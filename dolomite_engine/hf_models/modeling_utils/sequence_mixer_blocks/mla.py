import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import divide_if_divisible, is_flash_attention_available
from ...enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ..linear import ParameterizedLinear
from ..position_embedding import apply_rotary_pos_emb
from .softmax_attention import repeat_key_value


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        query_compression_size: int,
        key_value_compression_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: InitMethod,
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
        self.query_compression_size = query_compression_size
        self.key_value_compression_size = key_value_compression_size

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.attention_head_type = attention_head_type
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        if self.attention_head_type == AttentionHeadType.mha:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.num_heads

            assert (
                self.num_heads == self.num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"
        elif self.attention_head_type == AttentionHeadType.gqa:
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            divide_if_divisible(
                self.num_heads,
                self.num_key_value_heads,
                f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` ({self.num_key_value_heads})",
            )
        elif self.attention_head_type == AttentionHeadType.mqa:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, f"{self.__class__.__name__} should have 1 head for keys and values"
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_attn = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            bias=self.add_bias,
            std=std,
        )

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(self.hidden_size, self.hidden_size, bias=self.add_bias, std=std)

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

    def _prepare_qkv_for_forward(
        self, hidden_states: torch.Tensor, past_key_values: DynamicCache | None, rope_cos_sin: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # the output of following is a tuple if using MQA with tensor parallel
        hidden_states = self.c_attn_down_projection(hidden_states)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            compressed_query, compressed_key_value, key_rope = hidden_states.split(
                (
                    self.query_compression_size,
                    2 * self.key_value_compression_size,
                    self.rope_dim,
                ),
                dim=-1,
            )
            del hidden_states

            query_query_rope = self.query_up_projection(compressed_query)
            query, query_rope = query_query_rope.split((self.hidden_size, self.rope_dim), dim=-1)
            del query_query_rope

            query_rope = apply_rotary_pos_emb(query_rope, rope_cos_sin)
            query = torch.cat([query, query_rope], dim=-1)
            del query_rope

            key_rope = apply_rotary_pos_emb(key_rope, rope_cos_sin)

            if past_key_values is not None:
                compressed_key_value, key_rope = past_key_values.update(compressed_key_value, key_rope, self.layer_idx)

            key_value = self.key_value_up_projection(compressed_key_value)
            key, value = key_value.chunk(2, dim=-1)
            del key_value

            key = torch.cat([key, key_rope], dim=-1)
            del key_rope
        else:
            compressed_query, compressed_key, compressed_value = hidden_states.split(
                (self.query_compression_size, 2 * self.key_value_compression_size), dim=-1
            )
            del hidden_states

            query = self.query_up_projection(compressed_query)

            if past_key_values is not None:
                compressed_key, compressed_value = past_key_values.update(
                    compressed_key, compressed_value, self.layer_idx
                )

            key_value = self.key_value_up_projection(compressed_key_value)
            key, value = key_value.chunk(2, dim=-1)
            del key_value

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

        if self.position_embedding_type == PositionEmbeddingType.rope:
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
                if self.attention_head_type == AttentionHeadType.mqa:
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

    def _get_softmax_scale(self, return_none_allowed: bool = True) -> float:
        if self.attention_multiplier is None:
            softmax_scale = None if return_none_allowed else 1 / self.head_dim**0.5
        else:
            softmax_scale = self.attention_multiplier

        return softmax_scale
