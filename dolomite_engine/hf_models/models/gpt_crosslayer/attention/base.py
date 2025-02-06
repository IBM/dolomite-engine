import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import ParameterizedLinear, apply_rotary_pos_emb, get_normalization_function
from ..config import GPTCrossLayerConfig


class CrossLayerAttention(nn.Module):
    def __init__(self, config: GPTCrossLayerConfig, causal: bool, layer_idx: int | None = None) -> None:
        super().__init__()

        self.causal = causal
        self.mask_value = None
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.num_key_value_heads = config.num_key_value_heads
        self.add_bias = config.add_bias

        assert (
            self.hidden_size % self.num_heads == 0
        ), f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.hidden_size // self.num_heads
        self.attention_head_type = AttentionHeadType(config.attention_head_type)

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self.scale_attn_weights = config.scale_attn_weights
        self.attention_multiplier = config.attention_multiplier

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32

        self.q_attn = ParameterizedLinear(
            self.hidden_size, self.hidden_size, bias=self.add_bias, std=config.initializer_range
        )
        self.c_proj = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.add_bias,
            std=(config.initializer_range / math.sqrt(2 * config.num_layers)),
        )

        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop

        self.attn_dropout = nn.Identity() if self.attn_pdrop == 0 else nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Identity() if self.resid_pdrop == 0 else nn.Dropout(self.resid_pdrop)

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
        batch_size, query_length = hidden_states.shape[:2]

        query = self.q_attn(hidden_states)
        query = query.view(batch_size, query_length, self.num_heads, -1)
        query = query.transpose(1, 2)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)

        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype

        batch_size = query.shape[0]
        query_length = query.shape[2]
        key_length = key.shape[-1]

        # Always copies
        query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)

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

        hidden_states = F.softmax(hidden_states.to(softmax_dtype), dim=-1).to(dtype)
        hidden_states = self.attn_dropout(hidden_states)

        hidden_states = torch.matmul(hidden_states, value)

        del value

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.resid_dropout(hidden_states)

        return hidden_states

    def _get_softmax_scale(self, return_none_allowed: bool = True) -> float:
        if self.scale_attn_weights:
            if self.attention_multiplier is None:
                softmax_scale = None if return_none_allowed else 1 / self.head_dim**0.5
            else:
                softmax_scale = self.attention_multiplier
        else:
            softmax_scale = 1

        return softmax_scale


class KeyValueProjection(nn.Module):
    def __init__(self, config: GPTCrossLayerConfig) -> None:
        super().__init__()

        self.num_heads = config.n_head
        self.num_key_value_heads = config.num_key_value_heads

        head_dim = config.hidden_size // self.num_heads

        self.ln = get_normalization_function(
            config.normalization_function, config.hidden_size, config.layer_norm_epsilon
        )
        self.kv_attn = ParameterizedLinear(
            config.hidden_size,
            2 * self.num_key_value_heads * head_dim,
            bias=config.add_bias,
            std=config.initializer_range,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:2]

        hidden_states = self.ln(hidden_states)
        hidden_states = self.kv_attn(hidden_states)

        if self.num_key_value_heads == 1:
            hidden_states = hidden_states.unsqueeze(1)
        else:
            hidden_states = hidden_states.view(batch_size, query_length, self.num_key_value_heads, -1)
            hidden_states = hidden_states.transpose(1, 2)

        key, value = hidden_states.chunk(2, -1)

        return key, value
