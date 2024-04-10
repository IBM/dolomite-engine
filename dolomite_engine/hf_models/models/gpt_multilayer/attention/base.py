import math

import torch
import torch.nn as nn

from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import ParameterizedLinear, get_normalization_function
from ..config import GPTMultiLayerConfig


class MultiLayerAttention(nn.Module):
    def __init__(self, config: GPTMultiLayerConfig, causal: bool, layer_idx: int = None) -> None:
        super().__init__()

        self.causal = causal
        self.mask_value = None
        self.hidden_size = config.n_embd
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
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )

        self.q_attn = ParameterizedLinear(
            self.hidden_size, self.hidden_size, bias=self.add_bias, std=config.initializer_range
        )
        self.c_proj = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.add_bias,
            std=(config.initializer_range / math.sqrt(2 * config.n_layer)),
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

    def _get_mask_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(torch.float16).min, dtype=dtype, device=device)
        return self.mask_value


class KeyValueProjection(nn.Module):
    def __init__(self, config: GPTMultiLayerConfig) -> None:
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
