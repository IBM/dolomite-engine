import math
from typing import Tuple

import torch
import torch.distributed
import torch.nn as nn

from ....utils import ProcessGroupManager, SafeTensorsWeightsManager, get_cuda_rng_tracker
from ...config import CommonConfig
from ...enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ...modeling_utils import Attention, ParameterizedLinear
from ...utils import divide_if_divisible
from ..dropout import Dropout_TP
from ..linear import ColumnParallelLinear, RowParallelLinear
from ..TP import copy_to_tensor_parallel_region


class Attention_TP(Attention):
    def __init__(self, config: CommonConfig, causal: bool, layer_idx: int = None) -> None:
        nn.Module.__init__(self)

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

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
            self.global_hidden_size, tp_world_size, "hidden_size should be divisible by TP world size"
        )

        self.num_heads = divide_if_divisible(
            self.global_num_heads, tp_world_size, "num_heads must be divisible by TP world size"
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

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

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
                std=std,
            )
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
                tp_world_size,
                f"`num_key_value_heads` ({self.global_num_key_value_heads}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
            )

            self.c_attn = ColumnParallelLinear(
                self.global_hidden_size,
                self.global_hidden_size + 2 * self.global_num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )
        elif self.attention_head_type == AttentionHeadType.mqa:
            if self.global_num_key_value_heads is None:
                self.global_num_key_value_heads = 1

            assert (
                self.global_num_key_value_heads == 1
            ), f"{self.__class__.__name__} should have 1 head for keys and values"

            self.num_key_value_heads = 1

            self.c_attn = _MQA_QueryKeyValueProjection(config)
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = RowParallelLinear(self.global_hidden_size, self.global_hidden_size, bias=self.add_bias, std=std)

        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop

        self.attn_dropout = nn.Identity() if self.attn_pdrop == 0 else Dropout_TP(self.attn_pdrop)
        self.resid_dropout = nn.Identity() if self.resid_pdrop == 0 else Dropout_TP(self.resid_pdrop)

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        self.c_attn.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix=prefix + "c_attn.")
        self.c_proj.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix=prefix + "c_proj.")

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


class _MQA_QueryKeyValueProjection(nn.Module):
    def __init__(self, config: CommonConfig) -> None:
        super().__init__()

        self.global_hidden_size = config.n_embd
        self.add_bias = config.add_bias
        global_num_heads = config.n_head

        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer
        init_method = InitMethod(config.init_method)

        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        hidden_size = divide_if_divisible(
            self.global_hidden_size, self.tp_world_size, "hidden_size should be divisible by TP world size"
        )

        num_heads = divide_if_divisible(
            global_num_heads, self.tp_world_size, "num_heads must be divisible by TP world size"
        )
        self.head_dim = divide_if_divisible(hidden_size, num_heads, "")

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.q_attn = ColumnParallelLinear(
            self.global_hidden_size, self.global_hidden_size, bias=self.add_bias, std=std
        )

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.kv_attn = _TensorParallelSharedLinear(
            self.global_hidden_size, 2 * self.head_dim, bias=self.add_bias, std=std
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.q_attn(hidden_states)

        key_value = self.kv_attn(hidden_states)
        key_value = copy_to_tensor_parallel_region(key_value)
        key, value = key_value.chunk(2, -1)

        return query, key, value

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        tp_rank = ProcessGroupManager.get_tensor_parallel_rank()

        hidden_size_per_rank = divide_if_divisible(self.global_hidden_size, self.tp_world_size, "")
        start_index = tp_rank * hidden_size_per_rank
        end_index = (tp_rank + 1) * hidden_size_per_rank

        weight = safetensors_weight_manager.get_slice(prefix + "weight")
        q_attn_state_dict = {"weight": weight[start_index:end_index, :]}
        kv_attn_state_dict = {
            "weight": weight[self.global_hidden_size : self.global_hidden_size + 2 * self.head_dim, :]
        }

        if self.add_bias:
            bias = safetensors_weight_manager.get_slice(prefix + "bias")
            q_attn_state_dict["bias"] = bias[start_index:end_index]
            kv_attn_state_dict["bias"] = bias[self.global_hidden_size : self.global_hidden_size + 2 * self.head_dim]

        self.q_attn.load_state_dict(q_attn_state_dict)
        self.kv_attn.load_state_dict(kv_attn_state_dict)


class _TensorParallelSharedLinear(ParameterizedLinear):
    @torch.no_grad()
    def reset_parameters(self) -> None:
        with get_cuda_rng_tracker().fork():
            return super().reset_parameters()
