import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from .....utils import divide_if_divisible
from ....enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ...linear import ParameterizedLinear
from ...position_embedding import apply_rotary_pos_emb
from ..softmax_attention import repeat_key_value


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
    ) -> None:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias

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

        if init_method == InitMethod.mup:
            raise NotImplementedError("implementation needs to be checked for mup")

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_attn_down_projection = ParameterizedLinear(
            self.hidden_size,
            self.query_compression_size + 2 * self.key_value_compression_size,
            bias=self.add_bias,
            std=std,
        )

        # TODO add mup
        self.query_up_projection = ParameterizedLinear(
            self.query_compression_size,
            (1 + (self.position_embedding_type == PositionEmbeddingType.rope)) * self.num_heads * self.head_dim,
            bias=self.add_bias,
            std=std,
        )

        # TODO add mup
        self.key_value_up_projection = ParameterizedLinear(
            self.key_value_compression_size,
            self.num_key_value_heads * self.head_dim,
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

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_heads, -1)
        hidden_states = hidden_states.transpose(1, 2)

        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_key_value_heads, -1)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(batch_size, query_length, -1, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        query, key, value = hidden_states.split((self.hidden_size, self.head_dim, self.head_dim), dim=-1)

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
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        # the output of following is a tuple if using MQA with tensor parallel
        hidden_states = self.c_attn_down_projection(hidden_states)

        compressed_query, compressed_key_value = hidden_states.split(
            (self.query_compression_size, 2 * self.key_value_compression_size), dim=-1
        )

        query = self.query_up_projection(compressed_query)

        # TODO write a fused kernel for this to avoid copy
        if self.position_embedding_type == PositionEmbeddingType.rope:
            query, query_rope = query.chunk(2, dim=-1)
            query_rope = apply_rotary_pos_emb(query_rope, rope_cos_sin)
            query = torch.cat([query, query_rope], dim=-1)

        if past_key_values is not None:
            compressed_key_value = past_key_values.update(compressed_key_value, key_rope, self.layer_idx)

        key_value = self.key_value_up_projection(compressed_key_value)

        key, value = key_value.chunk(2, dim=-1)

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, [num_heads + num_key_value_heads * 2] * head_dim)
        # ==========================================================================================

        # for MHA, we can get away with doing just 1 transpose which is not true for GQA
        if self.attention_head_type == AttentionHeadType.mha:
            query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
        elif self.attention_head_type == AttentionHeadType.gqa:
            query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
        elif self.attention_head_type == AttentionHeadType.mqa:
            query, key, value = self._prepare_qkv_for_forward_mqa(hidden_states)
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

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
