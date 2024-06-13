from typing import Tuple

import torch
import torch.nn as nn

from ....utils import SafeTensorsWeightsManager
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import Attention, ParameterizedLinear
from ..dropout import Dropout_TP
from ..TP import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_to_tensor_parallel_region,
    get_tensor_parallel_group_manager,
)


class Attention_TP(Attention):
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
        attention_multiplier: float,
        attention_softmax_in_fp32: bool,
        scale_attention_softmax_in_fp32: bool,
        attn_pdrop: float,
        resid_pdrop: float,
        layer_idx: int = None,
    ) -> None:
        nn.Module.__init__(self)

        self.tp_rank = get_tensor_parallel_group_manager().get_rank()
        self.tp_world_size = get_tensor_parallel_group_manager().get_world_size()

        self.causal = causal
        self.mask_value = None
        self.add_bias = add_bias

        self.global_hidden_size = hidden_size
        self.hidden_size = self.global_hidden_size // self.tp_world_size

        self.global_num_heads = num_attention_heads
        self.num_heads = num_attention_heads // self.tp_world_size

        self.global_num_key_value_heads = num_key_value_heads
        self.num_key_value_heads = num_key_value_heads // self.tp_world_size

        assert self.global_num_heads % self.tp_world_size == 0, "num_heads must be divisible by TP world size"
        assert (
            self.global_hidden_size % self.global_num_heads == 0
        ), f"`embed_dim` ({self.global_hidden_size}) must be divisible by `num_heads` ({self.global_num_heads})"

        self.head_dim = self.hidden_size // self.num_heads
        self.attention_head_type = attention_head_type

        self.position_embedding_type = position_embedding_type
        self.scale_attn_weights = scale_attention_weights
        self.attention_multiplier = attention_multiplier

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32 and attention_softmax_in_fp32

        if self.attention_head_type == AttentionHeadType.mha:
            if self.global_num_key_value_heads is None:
                self.global_num_key_value_heads = self.global_num_heads

            assert (
                self.global_num_heads == self.global_num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"

            self.num_key_value_heads = self.num_heads

            self.c_attn = ColumnParallelLinear(
                self.global_hidden_size,
                self.global_hidden_size + 2 * self.global_num_key_value_heads * self.head_dim,
                bias=self.add_bias,
            )
        elif self.attention_head_type == AttentionHeadType.gqa:
            assert (
                self.global_num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert self.global_num_key_value_heads > 1, (
                "GroupedQueryAttention should have more than 1 head for keys and values, use MultiQueryAttention class if "
                "you want to use 1 head for keys and values"
            )

            assert self.global_num_heads % self.global_num_key_value_heads == 0, (
                f"`num_heads` ({self.global_num_heads}) should be a multiple of `num_key_value_heads` "
                f"({self.global_num_key_value_heads})"
            )

            assert self.global_num_key_value_heads % self.tp_world_size == 0, (
                f"`num_key_value_heads` ({self.global_num_key_value_heads}) must be divisible by `tensor_parallel_world_size` "
                f"({self.tp_world_size})"
            )

            self.num_key_value_heads = self.global_num_key_value_heads // self.tp_world_size

            self.c_attn = ColumnParallelLinear(
                self.global_hidden_size,
                self.global_hidden_size + 2 * self.global_num_key_value_heads * self.head_dim,
                bias=self.add_bias,
            )
        elif self.attention_head_type == AttentionHeadType.mqa:
            if self.global_num_key_value_heads is None:
                self.global_num_key_value_heads = 1

            assert (
                self.global_num_key_value_heads == 1
            ), f"{self.__class__.__name__} should have 1 head for keys and values"

            self.num_key_value_heads = 1

            self.c_attn = _MQA_KeyValueProjection(self.global_hidden_size, self.head_dim, add_bias=self.add_bias)

        self.c_proj = RowParallelLinear(self.global_hidden_size, self.global_hidden_size, bias=self.add_bias)

        self.attn_pdrop = attn_pdrop

        self.attn_dropout = Dropout_TP(self.attn_pdrop)
        self.resid_dropout = Dropout_TP(resid_pdrop)

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
        self.c_attn.load_unsharded_weights(
            safetensors_weight_manager,
            prefix=prefix if self.attention_head_type == AttentionHeadType.mqa else prefix + "c_attn.",
        )
        self.c_proj.load_unsharded_weights(safetensors_weight_manager, prefix=prefix + "c_proj.")

    def _prepare_qkv_for_forward_mqa(
        self, query_key_value: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = query_key_value
        batch_size, query_length = query.shape[:-1]

        query = query.view(batch_size, query_length, self.num_heads, -1)

        query = query.transpose(1, 2)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value


class _MQA_KeyValueProjection(nn.Module):
    def __init__(self, global_hidden_size: int, head_dim: int, add_bias: bool) -> None:
        super().__init__()

        self.global_hidden_size = global_hidden_size
        self.head_dim = head_dim
        self.add_bias = add_bias

        self.tp_rank = get_tensor_parallel_group_manager().get_rank()
        self.tp_world_size = get_tensor_parallel_group_manager().get_world_size()

        self.q_attn = ColumnParallelLinear(global_hidden_size, global_hidden_size, bias=add_bias)
        self.kv_attn = ParameterizedLinear(global_hidden_size, 2 * head_dim, bias=add_bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.q_attn(hidden_states)

        key_value = self.kv_attn(hidden_states)
        key_value = copy_to_tensor_parallel_region(key_value)
        key, value = key_value.chunk(2, -1)

        return query, key, value

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
        start_index = self.tp_rank * (self.global_hidden_size // self.tp_world_size)
        end_index = (self.tp_rank + 1) * (self.global_hidden_size // self.tp_world_size)

        weight = safetensors_weight_manager.get_slice(prefix + "c_attn.weight")
        q_attn_state_dict = {"weight": weight[start_index:end_index, :]}
        kv_attn_state_dict = {
            "weight": weight[self.global_hidden_size : self.global_hidden_size + 2 * self.head_dim, :]
        }

        if self.add_bias:
            bias = safetensors_weight_manager.get_slice(prefix + "c_attn.bias")
            q_attn_state_dict["bias"] = bias[start_index:end_index]
            kv_attn_state_dict["bias"] = bias[self.global_hidden_size : self.global_hidden_size + 2 * self.head_dim]

        self.q_attn.load_state_dict(q_attn_state_dict)
        self.kv_attn.load_state_dict(kv_attn_state_dict)
