import torch
import torch.nn as nn

from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import get_normalization_function
from ..config import GPTMultiLayerConfig


class MultiLayerAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        causal: bool,
        add_bias: bool,
        scale_attention_weights: bool,
        attention_softmax_in_fp32: bool,
        scale_attention_softmax_in_fp32: bool,
        attn_pdrop: float,
        resid_pdrop: float,
        layer_idx: int = None,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.mask_value = None
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias

        assert (
            self.hidden_size % self.num_heads == 0
        ), f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.hidden_size // self.num_heads
        self.attention_head_type = attention_head_type

        self.position_embedding_type = position_embedding_type
        self.scale_attn_weights = scale_attention_weights

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32 and attention_softmax_in_fp32

        self.q_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=self.add_bias)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.add_bias)

        self.attn_pdrop = attn_pdrop

        self.attn_dropout = nn.Identity() if attn_pdrop == 0 else nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Identity() if resid_pdrop == 0 else nn.Dropout(resid_pdrop)

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
        self.kv_attn = nn.Linear(config.hidden_size, 2 * self.num_key_value_heads * head_dim, bias=config.add_bias)

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
