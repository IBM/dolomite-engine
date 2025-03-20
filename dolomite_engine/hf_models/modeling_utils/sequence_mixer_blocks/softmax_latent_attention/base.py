import math

import torch
import torch.nn as nn

from .....utils import divide_if_divisible
from ....enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ...linear import ParameterizedLinear
from ..softmax_attention import Attention


class MLAAttention(Attention):
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
        init_method: InitMethod,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        kv_compression_dim: int = None,
        use_latent_attention: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias

        self.kv_compression_dim = kv_compression_dim
        self.use_latent_attention = use_latent_attention

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

        # add the latent attention layer here
        if self.use_latent_attention:
            assert (
                self.kv_compression_dim is not None
            ), "kv_compression_dim must be specified when using latent attention"

            # For latent attention, we have separate projections
            self.q_proj = ParameterizedLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )

            self.kv_down_proj = ParameterizedLinear(
                self.hidden_size,
                self.kv_compression_dim,
                bias=self.add_bias,
                std=std,
            )

            self.k_up_proj = ParameterizedLinear(
                self.kv_compression_dim,
                self.num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )

            self.v_up_proj = ParameterizedLinear(
                self.kv_compression_dim,
                self.num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )
        else:
            # Original implementation for non-latent attention
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

    ## TODO: Need to verify if the current implementation is correct or not.
    def _prepare_qkv_for_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        if self.use_latent_attention:
            # Latent attention flow
            query = self.q_proj(hidden_states)
            kv_latent = self.kv_down_proj(hidden_states)
            key = self.k_up_proj(kv_latent)
            value = self.v_up_proj(kv_latent)

            batch_size, query_length = hidden_states.shape[:-1]

            # Reshape query
            query = query.view(batch_size, query_length, self.num_heads, self.head_dim)
            query = query.transpose(1, 2)

            # Reshape key and value based on attention head type
            if self.attention_head_type == AttentionHeadType.mha:
                key = key.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
                value = value.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
            elif self.attention_head_type == AttentionHeadType.gqa:
                key = key.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
                value = value.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
            elif self.attention_head_type == AttentionHeadType.mqa:
                key = key.view(batch_size, query_length, 1, self.head_dim)
                value = value.view(batch_size, query_length, 1, self.head_dim)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
        else:
            # Original attention flow
            hidden_states = self.c_attn(hidden_states)

            # Use existing methods based on attention head type
            if self.attention_head_type == AttentionHeadType.mha:
                query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
            elif self.attention_head_type == AttentionHeadType.gqa:
                query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
            elif self.attention_head_type == AttentionHeadType.mqa:
                query, key, value = self._prepare_qkv_for_forward_mqa(hidden_states)
            else:
                raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================
        return query, key, value
