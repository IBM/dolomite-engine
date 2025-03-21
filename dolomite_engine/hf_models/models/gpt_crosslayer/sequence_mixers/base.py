import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .....enums import Kernel
from .....kernels import is_kernel_allowed
from .....utils import divide_if_divisible, is_flash_attention_available
from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import ParameterizedLinear, apply_rotary_pos_emb, get_normalization_function


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class CrossLayerAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        initializer_range: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.mask_value = None
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias
        self.use_padding_free_transformer = use_padding_free_transformer

        assert (
            self.hidden_size % self.num_heads == 0
        ), f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.hidden_size // self.num_heads
        self.attention_head_type = attention_head_type

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier

        self.layer_idx = layer_idx

        self.q_attn = ParameterizedLinear(
            self.hidden_size, self.hidden_size, bias=self.add_bias, std=initializer_range
        )
        self.c_proj = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.add_bias,
            std=initializer_range / math.sqrt(2 * num_layers),
        )

        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

        assert (
            self.num_key_value_heads is not None
        ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

        assert self.num_heads % self.num_key_value_heads == 0, (
            f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` "
            f"({self.num_key_value_heads})"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_kernel_allowed(Kernel.flash_attention_2):
            if self.use_padding_free_transformer:
                total_q = hidden_states.shape[0]

                query = self.q_attn(hidden_states)
                query = query.view(total_q, self.num_heads, -1)

                if self.position_embedding_type == PositionEmbeddingType.rope:
                    query = apply_rotary_pos_emb(query, rope_cos_sin)

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

                del query, key, value

                hidden_states = hidden_states.view(-1, self.hidden_size)
            else:
                batch_size, query_length = hidden_states.shape[:2]

                query = self.q_attn(hidden_states)
                query = query.view(batch_size, query_length, self.num_heads, -1)

                if self.position_embedding_type == PositionEmbeddingType.rope:
                    # TODO avoid this extra transpose
                    query = query.transpose(1, 2)
                    query = apply_rotary_pos_emb(query, rope_cos_sin)
                    query = query.transpose(1, 2)

                batch_size, query_length = query.shape[:2]

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

                hidden_states = hidden_states.view(batch_size, query_length, -1)
        else:
            batch_size, query_length = hidden_states.shape[:2]

            query = self.q_attn(hidden_states)
            query = query.view(batch_size, query_length, self.num_heads, -1)
            query = query.transpose(1, 2)

            if self.position_embedding_type == PositionEmbeddingType.rope:
                query = apply_rotary_pos_emb(query, rope_cos_sin)

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

            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def _get_softmax_scale(self, return_none_allowed: bool = True) -> float:
        if self.attention_multiplier is None:
            softmax_scale = None if return_none_allowed else 1 / self.head_dim**0.5
        else:
            softmax_scale = self.attention_multiplier

        return softmax_scale


class KeyValueProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        add_bias: bool,
        initializer_range: float,
        normalization_function: str,
        layer_norm_epsilon: float,
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__()

        self.num_key_value_heads = num_key_value_heads
        head_dim = divide_if_divisible(hidden_size, num_attention_heads, "")

        self.ln = get_normalization_function(normalization_function, hidden_size, layer_norm_epsilon)
        self.kv_attn = ParameterizedLinear(
            hidden_size,
            2 * self.num_key_value_heads * head_dim,
            bias=add_bias,
            std=initializer_range,
        )

        self.use_padding_free_transformer = use_padding_free_transformer

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.ln(hidden_states)
        hidden_states = self.kv_attn(hidden_states)

        if self.use_padding_free_transformer:
            total_q = hidden_states.shape[0]

            if self.num_key_value_heads == 1:
                hidden_states = hidden_states.unsqueeze(1)
            else:
                hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)
        else:
            batch_size, query_length = hidden_states.shape[:2]

            if self.num_key_value_heads == 1:
                hidden_states = hidden_states.unsqueeze(1)
            else:
                hidden_states = hidden_states.view(batch_size, query_length, self.num_key_value_heads, -1)
                hidden_states = hidden_states.transpose(1, 2)

        key, value = hidden_states.chunk(2, -1)

        return key, value
