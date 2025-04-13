import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .....enums import Kernel
from .....kernels import is_kernel_allowed, wait_for_ACT
from .....utils import divide_if_divisible, is_flash_attention_available
from ...linear import ParameterizedLinear
from ...position_embedding import apply_rotary_pos_emb
from .utils import (
    interleave_query_key_value_tensor_for_gqa,
    interleave_query_key_value_tensor_for_mha,
    interleave_query_key_value_tensor_for_mqa,
    repeat_key_value,
    split_query_key_value_tensor_for_gqa,
    split_query_key_value_tensor_for_mha,
    split_query_key_value_tensor_for_mqa,
)


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        position_embedding_type: str,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias
        self.use_padding_free_transformer = use_padding_free_transformer

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
        )

        if num_attention_heads == num_key_value_heads:
            self.attention_head_type = "mha"
        elif num_key_value_heads == 1:
            self.attention_head_type = "mqa"
        else:
            self.attention_head_type = "gqa"

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        if self.attention_head_type == "mha":
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.num_heads

            assert (
                self.num_heads == self.num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"
        elif self.attention_head_type == "gqa":
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            divide_if_divisible(
                self.num_heads,
                self.num_key_value_heads,
                f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` ({self.num_key_value_heads})",
            )
        elif self.attention_head_type == "mqa":
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, f"{self.__class__.__name__} should have 1 head for keys and values"
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_attn = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            bias=self.add_bias,
            std=std,
        )

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(self.hidden_size, self.hidden_size, bias=self.add_bias, std=std)

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

    def _prepare_qkv_for_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # the output of following is a tuple if using MQA with tensor parallel
        hidden_states = self.c_attn(hidden_states)

        # for MHA, we can get away with doing just 1 transpose which is not true for GQA
        if self.attention_head_type == "mha":
            query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
        elif self.attention_head_type == "gqa":
            query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
        elif self.attention_head_type == "mqa":
            query, key, value = self._prepare_qkv_for_forward_mqa(hidden_states)
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        return query, key, value

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_padding_free_transformer:
            total_q = hidden_states.shape[0]
            hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)
        else:
            batch_size, query_length = hidden_states.shape[:-1]
            hidden_states = hidden_states.view(batch_size, query_length, self.num_heads, -1)
            hidden_states = hidden_states.transpose(1, 2)

        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_padding_free_transformer:
            total_q = hidden_states.shape[0]
            input_shape = (hidden_states.shape[0], self.num_key_value_heads, -1)
            output_shape = (total_q, -1, self.head_dim)
        else:
            batch_size, query_length = hidden_states.shape[:-1]
            input_shape = (*hidden_states.shape[:-1], self.num_key_value_heads, -1)
            output_shape = (batch_size, query_length, -1, self.head_dim)

        hidden_states = hidden_states.view(*input_shape)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        query = query.reshape(*output_shape)

        if not self.use_padding_free_transformer:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_padding_free_transformer:
            total_q = hidden_states.shape[0]
        else:
            batch_size, query_length = hidden_states.shape[:-1]

        query, key, value = hidden_states.split((self.hidden_size, self.head_dim, self.head_dim), dim=-1)

        if self.use_padding_free_transformer:
            query = query.view(total_q, self.num_heads, -1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        else:
            query = query.view(batch_size, query_length, self.num_heads, -1)

            query = query.transpose(1, 2)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert is_kernel_allowed(Kernel.flash_attention_2)
            assert past_key_values is None

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        if self.position_embedding_type == "rope":
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        if is_kernel_allowed(Kernel.flash_attention_2):
            if self.use_padding_free_transformer:
                output_shape = (-1, self.hidden_size)
            else:
                # TODO avoid this extra transpose
                query = query.transpose(1, 2)
                if self.attention_head_type == "mqa":
                    key = key.squeeze(1).unsqueeze(2)
                    value = value.squeeze(1).unsqueeze(2)
                else:
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)

                batch_size, query_length = query.shape[:2]
                output_shape = (batch_size, query_length, -1)

            query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
            key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
            value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

            if self.use_padding_free_transformer:
                hidden_states = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=self.softmax_dropout_p if self.training else 0,
                    softmax_scale=self._get_softmax_scale(),
                    causal=self.causal,
                )
            else:
                hidden_states = _flash_attention_forward(
                    query_states=query,
                    key_states=key,
                    value_states=value,
                    attention_mask=attention_mask,
                    query_length=query_length,
                    is_causal=self.causal,
                    dropout=self.softmax_dropout_p if self.training else 0,
                    softmax_scale=self._get_softmax_scale(),
                )

            del query, key, value

            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
            hidden_states = hidden_states.view(*output_shape)
        else:
            key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
            value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.softmax_dropout_p if self.training else 0,
                is_causal=self.causal if attention_mask is None else False,
                scale=self._get_softmax_scale(),
            )

            del query, key, value

            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def _get_softmax_scale(self) -> float:
        if self.attention_multiplier is None:
            softmax_scale = None
        else:
            softmax_scale = self.attention_multiplier

        return softmax_scale


_INTERLEAVE_FUNCTIONS = {
    "mha": interleave_query_key_value_tensor_for_mha,
    "mqa": interleave_query_key_value_tensor_for_mqa,
    "gqa": interleave_query_key_value_tensor_for_gqa,
}


_SPLIT_FUNCTIONS = {
    "mha": split_query_key_value_tensor_for_mha,
    "mqa": split_query_key_value_tensor_for_mqa,
    "gqa": split_query_key_value_tensor_for_gqa,
}


def interleave_query_key_value_tensor_for_attention(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_head_type: str,
) -> torch.Tensor:
    if attention_head_type in _INTERLEAVE_FUNCTIONS:
        interleave_function = _INTERLEAVE_FUNCTIONS[attention_head_type]
        interleave_function_parameters = inspect.signature(interleave_function).parameters.keys()

        parameters_to_pass = {}
        this_function_parameters = locals()
        for parameter in interleave_function_parameters:
            parameters_to_pass[parameter] = this_function_parameters[parameter]

        query_key_value_weight = interleave_function(**parameters_to_pass)

        return query_key_value_weight

    raise ValueError(f"unexpected `attention_head_type` {attention_head_type}")


def split_query_key_value_tensor_for_attention(
    query_key_value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_head_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if attention_head_type in _SPLIT_FUNCTIONS:
        split_function = _SPLIT_FUNCTIONS[attention_head_type]
        split_function_parameters = inspect.signature(split_function).parameters.keys()

        parameters_to_pass = {}
        this_function_parameters = locals()
        for parameter in split_function_parameters:
            parameters_to_pass[parameter] = this_function_parameters[parameter]

        return split_function(**parameters_to_pass)

    raise ValueError(f"unexpected `attention_head_type` {attention_head_type}")
