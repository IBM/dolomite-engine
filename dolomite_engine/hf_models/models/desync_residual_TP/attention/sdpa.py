import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from .....utils import ProcessGroupManager, divide_if_divisible
from ....enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ....modeling_utils import SDPA, apply_rotary_pos_emb, repeat_key_value
from ....modeling_utils_TP import Attention_TP, ColumnParallelLinear
from ...desync_residual import DesyncResidualConfig
from ..linear import DesyncResidualLinear_TP, DesyncResidualRowParallelLinear


class DesyncResidualSDPA_TP(Attention_TP, SDPA):
    def __init__(self, config: DesyncResidualConfig, causal: bool, layer_idx: int = None) -> None:
        nn.Module.__init__(self)

        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.causal = causal
        self.global_hidden_size = config.hidden_size
        self.global_num_heads = config.num_attention_heads
        self.global_num_key_value_heads = config.num_key_value_heads
        self.add_bias = config.add_bias
        self.m_residual = config.m_residual

        self.previous_mlp_all_reduce = layer_idx == 0 or config.reduce_pattern[layer_idx - 1]["mlp"]
        self.current_attention_all_reduce = config.reduce_pattern[layer_idx]["attention"]

        initializer_range = config.initializer_range
        m_width = config.m_width
        num_layers = config.num_layers
        init_method = InitMethod(config.init_method)

        divide_if_divisible(
            self.global_hidden_size,
            self.global_num_heads,
            f"`embed_dim` ({self.global_hidden_size}) must be divisible by `num_heads` ({self.global_num_heads})",
        )

        self.hidden_size = divide_if_divisible(
            self.global_hidden_size, self.tp_world_size, "hidden_size should be divisible by TP world size"
        )

        self.num_heads = divide_if_divisible(
            self.global_num_heads, self.tp_world_size, "num_heads must be divisible by TP world size"
        )

        self.head_dim = divide_if_divisible(self.hidden_size, self.num_heads, "")
        self.attention_head_type = AttentionHeadType(config.attention_head_type)

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self.attention_multiplier = config.attention_multiplier

        self.layer_idx = layer_idx

        if self.attention_head_type == AttentionHeadType.mha:
            if self.global_num_key_value_heads is None:
                self.global_num_key_value_heads = self.global_num_heads

            assert (
                self.global_num_heads == self.global_num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"

            self.num_key_value_heads = self.num_heads
        elif self.attention_head_type == AttentionHeadType.gqa:
            assert (
                self.global_num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert self.global_num_key_value_heads > 1, (
                "GroupedQueryAttention should have more than 1 head for keys and values, use MultiQueryAttention class if "
                "you want to use 1 head for keys and values"
            )

            divide_if_divisible(
                self.global_num_heads,
                self.global_num_key_value_heads,
                f"`num_heads` ({self.global_num_heads}) should be a multiple of `num_key_value_heads` ({self.global_num_key_value_heads})",
            )

            self.num_key_value_heads = divide_if_divisible(
                self.global_num_key_value_heads,
                self.tp_world_size,
                f"`num_key_value_heads` ({self.global_num_key_value_heads}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
            )
        elif self.attention_head_type == AttentionHeadType.mqa:
            raise ValueError("mqa is not supported with DesyncResidualAttention")
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        # first layer needs and any attention after an mlp with all reduce needs column parallel
        if self.previous_mlp_all_reduce:
            self.c_attn = ColumnParallelLinear(
                self.global_hidden_size,
                self.global_hidden_size + 2 * self.global_num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )
        else:
            self.c_attn = DesyncResidualLinear_TP(
                self.global_hidden_size,
                self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        if self.current_attention_all_reduce:
            self.c_proj = DesyncResidualRowParallelLinear(
                self.global_hidden_size, self.global_hidden_size, bias=self.add_bias, std=std
            )
        else:
            self.c_proj = DesyncResidualLinear_TP(
                self.hidden_size, self.global_hidden_size, bias=self.add_bias, std=std
            )

        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop

        self.attn_dropout = nn.Identity() if self.attn_pdrop == 0 else nn.Dropout(self.attn_pdrop)

        assert self.resid_pdrop == 0, "residual dropout is not supported with DesyncResidual"
        self.resid_dropout = nn.Identity() if self.resid_pdrop == 0 else nn.Dropout(self.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_heads, key_length, head_dim)
        # value -> (batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

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

        # ==========================================================================================
        # hidden_states -> (batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        if self.current_attention_all_reduce:
            hidden_states = self.c_proj(hidden_states, residual)
        else:
            hidden_states = self.c_proj(hidden_states)
            hidden_states = hidden_states + residual

        hidden_states = self.resid_dropout(hidden_states)

        return hidden_states
