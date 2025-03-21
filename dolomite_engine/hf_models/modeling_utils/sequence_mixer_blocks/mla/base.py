import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from .....utils import divide_if_divisible
from ....enums import InitMethod, PositionEmbeddingType
from ...linear import ParameterizedLinear
from ...position_embedding import apply_rotary_pos_emb
from ..softmax_attention import repeat_key_value


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        compress_query: bool,
        query_compression_size: int,
        key_value_compression_size: int,
        rope_dim: int,
        num_attention_heads: int,
        attention_multiplier: float,
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
    ) -> None:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = self.num_heads
        self.add_bias = add_bias
        self.compress_query = compress_query

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.query_compression_size = query_compression_size
        self.key_value_compression_size = key_value_compression_size
        self.rope_dim = rope_dim

        if self.position_embedding_type == PositionEmbeddingType.rope:
            assert self.rope_dim != 0
        else:
            assert self.rope_dim == 0

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        if init_method == InitMethod.mup:
            raise NotImplementedError("implementation needs to be checked for mup")

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        self.c_attn_down_projection = ParameterizedLinear(
            self.hidden_size,
            self.query_compression_size + 2 * self.key_value_compression_size + self.rope_dim,
            bias=self.add_bias,
            std=std,
        )

        self.query_up_projection = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size + self.rope_dim,
            bias=self.add_bias,
            std=std,
        )

        self.key_value_up_projection = ParameterizedLinear(
            self.hidden_size,
            self.query_compression_size + 2 * self.key_value_compression_size + self.rope_dim,
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
            compressed_query, compressed_key_value = hidden_states.split(
                (self.query_compression_size, 2 * self.key_value_compression_size), dim=-1
            )
            del hidden_states

            query = self.query_up_projection(compressed_query)

            if past_key_values is not None:
                compressed_key_value, None = past_key_values.update(compressed_key_value, None, self.layer_idx)

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
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states, past_key_values, rope_cos_sin)

        key = key.transpose(-1, -2)
        dtype = query.dtype

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, head_dim, key_length)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        batch_size = query.shape[0]
        query_length = query.shape[2]
        key_length = key.shape[-1]

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # Always copies
        query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
        # No copy when layer_past is provided.
        key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        # ==========================================================================================
        # query -> (batch_size * num_heads, query_length, head_dim)
        # key -> (batch_size * num_heads, head_dim, key_length)
        # value -> (batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

        if attention_mask is None:
            hidden_states = torch.empty(
                (batch_size * self.num_heads, query_length, key_length), device=query.device, dtype=query.dtype
            )
            beta = 0
        else:
            hidden_states = attention_mask.expand(-1, self.num_heads, -1, -1).reshape(-1, query_length, key_length)
            beta = 1

        hidden_states = torch.baddbmm(hidden_states, query, key, beta=beta, alpha=self._get_softmax_scale(False)).view(
            batch_size, self.num_heads, query_length, key_length
        )

        del query, key

        # ==========================================================================================
        # hidden_states -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        hidden_states = F.softmax(hidden_states.float(), dim=-1).to(dtype)
        hidden_states = self.softmax_dropout(hidden_states)

        # ==========================================================================================
        # value -> (batch_size, num_heads, key_length, head_dim)
        # hidden_states -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        hidden_states = torch.matmul(hidden_states, value)

        del value

        # ==========================================================================================
        # hidden_states -> (batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def _get_softmax_scale(self, return_none_allowed: bool = True) -> float:
        if self.attention_multiplier is None:
            softmax_scale = None if return_none_allowed else 1 / self.head_dim**0.5
        else:
            softmax_scale = self.attention_multiplier

        return softmax_scale
