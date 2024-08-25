import math

import torch.nn as nn

from .....utils import ProcessGroupManager, SafeTensorsWeightsManager
from ....enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ....modeling_utils import ParameterizedLinear
from ....modeling_utils_TP import (
    Attention_TP,
    ColumnParallelLinear,
    RowParallelLinear,
    tensor_parallel_split_safetensor_slice,
)
from ....modeling_utils_TP.attention import Attention_TP
from ....utils import divide_if_divisible
from ...gpt_ensemble import GPTEnsembleConfig


class EnsembleAttention_TP(Attention_TP):
    def __init__(self, config: GPTEnsembleConfig, causal: bool, layer_idx: int = None) -> None:
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
                tp_world_size,
                f"`num_key_value_heads` ({self.global_num_key_value_heads}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
            )
        elif self.attention_head_type == AttentionHeadType.mqa:
            raise ValueError("mqa is not supported with EnsembleAttention")
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        # first layer needs and any attention after an mlp with all reduce needs column parallel
        if layer_idx == 0 or config.reduce_pattern[layer_idx - 1]["mlp"]:
            self.c_attn = ColumnParallelLinear(
                self.global_hidden_size,
                self.global_hidden_size + 2 * self.global_num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )
        else:
            self.c_attn = ParameterizedLinear(
                self.global_hidden_size,
                self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_attn = ColumnParallelLinear(
            self.global_hidden_size,
            self.global_hidden_size + 2 * self.global_num_key_value_heads * self.head_dim,
            bias=self.add_bias,
            std=std,
        )

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        if config.reduce_pattern[layer_idx]["attention"]:
            self.c_proj = RowParallelLinear(
                self.global_hidden_size, self.global_hidden_size, bias=self.add_bias, std=std
            )
        else:
            self.c_proj = ParameterizedLinear(self.hidden_size, self.global_hidden_size, bias=self.add_bias, std=std)

        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop

        self.attn_dropout = nn.Identity() if self.attn_pdrop == 0 else nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Identity() if self.resid_pdrop == 0 else nn.Dropout(self.resid_pdrop)

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        self.c_attn.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix=prefix + "c_attn.")

        weight = safetensors_weight_manager.get_slice(prefix + "c_proj.weight")
        weight = tensor_parallel_split_safetensor_slice(weight, dim=0)
        state = {"weight": weight}
        if self.add_bias:
            bias = safetensors_weight_manager.get_slice(prefix + "c_proj.bias")
            bias = tensor_parallel_split_safetensor_slice(bias, dim=0)
            state["bias"] = bias

        self.c_proj.load_state_dict(state)
