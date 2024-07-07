import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from ....enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ....modeling_utils import Attention, apply_rotary_pos_emb, repeat_key_value
from ....utils import divide_if_divisible
from ..config import GPTEnsembleConfig
from ..linear import EnsembleLinear


class EnsembleAttention(Attention):
    def __init__(self, config: GPTEnsembleConfig, causal: bool, layer_idx: int = None) -> None:
        nn.Module.__init__(self)

        self.tp_world_size = config.pretraining_tensor_parallel_size

        self.causal = causal
        self.global_hidden_size = config.n_embd
        self.global_num_heads = config.n_head
        self.global_num_key_value_heads = config.num_key_value_heads
        self.add_bias = config.add_bias

        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer
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
        self.scale_attn_weights = config.scale_attn_weights
        self.attention_multiplier = config.attention_multiplier

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )

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
            raise ValueError("mqa is not supported with EnsembleAttention")
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_attn = EnsembleLinear(
            self.global_hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            tensor_parallel_size=self.tp_world_size,
            bias=self.add_bias,
            std=std,
        )

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = EnsembleLinear(
            self.hidden_size,
            self.global_hidden_size,
            tensor_parallel_size=self.tp_world_size,
            bias=self.add_bias,
            std=std,
        )

        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop

        self.attn_dropout = nn.Identity() if self.attn_pdrop == 0 else nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Identity() if self.resid_pdrop == 0 else nn.Dropout(self.resid_pdrop)

    def _prepare_qkv_for_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # hidden_states -> (1, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        hidden_states = self.c_attn(hidden_states)

        # ==========================================================================================
        # hidden_states -> (TP, batch_size, query_length, [num_heads + num_key_value_heads * 2] * head_dim)
        # ==========================================================================================

        tp, batch_size, query_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(tp * batch_size, query_length, -1)

        # ==========================================================================================
        # hidden_states -> (TP * batch_size, query_length, [num_heads + num_key_value_heads * 2] * head_dim)
        # ==========================================================================================

        # for MHA, we can get away with doing just 1 transpose which is not true for GQA
        if self.attention_head_type == AttentionHeadType.mha:
            query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
        elif self.attention_head_type == AttentionHeadType.gqa:
            query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
        elif self.attention_head_type == AttentionHeadType.mqa:
            raise NotImplementedError("mqa doesn't work with GPTEnsemble")
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache = None,
        attention_mask: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> torch.Tensor:
        # ==========================================================================================
        # hidden_states -> (1, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (TP * batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        key = key.transpose(-1, -2)

        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, head_dim, key_length)
        # value -> (TP * batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        tp_times_batch_size = query.shape[0]
        query_length = query.shape[2]
        key_length = key.shape[-1]

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # Always copies
        query = query.reshape(tp_times_batch_size * self.num_heads, query_length, self.head_dim)
        # No copy when layer_past is provided.
        key = key.reshape(tp_times_batch_size * self.num_heads, self.head_dim, key_length)

        # ==========================================================================================
        # query -> (TP * batch_size * num_heads, query_length, head_dim)
        # key -> (TP * batch_size * num_heads, head_dim, key_length)
        # value -> (TP * batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

        if attention_mask is None:
            attn_weights = torch.empty(
                (tp_times_batch_size * self.num_heads, query_length, key_length),
                device=query.device,
                dtype=query.dtype,
            )
            beta = 0
        else:
            # TODO avoid these repeats on every layer
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
            attention_mask = attention_mask.repeat(self.tp_world_size, 1, 1, 1)
            attn_weights = attention_mask.view(-1, query_length, key_length)
            beta = 1

        attn_weights = torch.baddbmm(attn_weights, query, key, beta=beta, alpha=self._get_softmax_scale(False)).view(
            tp_times_batch_size, self.num_heads, query_length, key_length
        )

        # ==========================================================================================
        # attn_weights -> (TP * batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        attn_weights = F.softmax(attn_weights.to(softmax_dtype), dim=-1).to(dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # ==========================================================================================
        # value -> (TP * batch_size, num_heads, key_length, head_dim)
        # attn_weights -> (TP * batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        attn_output = torch.matmul(attn_weights, value)

        # ==========================================================================================
        # attn_output -> (TP * batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(self.tp_world_size, -1, query_length, self.num_heads * self.head_dim)

        # ==========================================================================================
        # attn_output -> (TP, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # ==========================================================================================
        # attn_output -> (TP, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        return attn_output