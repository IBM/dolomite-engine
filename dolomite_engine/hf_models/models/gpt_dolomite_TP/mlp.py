import math

import torch
import torch.nn as nn

from ....utils import ProcessGroupManager, SafeTensorsWeightsManager
from ...enums import InitMethod
from ...modeling_utils import get_activation_function, is_glu
from ...modeling_utils_TP import (
    ColumnParallelLinear,
    Dropout_TP,
    RowParallelLinear,
    tensor_parallel_split_safetensor_slice,
)
from ...utils import divide_if_divisible
from ..gpt_dolomite import GPTDolomiteConfig
from ..gpt_dolomite.mlp import MLP


class MLP_TP(MLP):
    def __init__(self, config: GPTDolomiteConfig) -> None:
        nn.Module.__init__(self)

        hidden_size = config.n_embd
        intermediate_size = config.n_inner
        activation_function = config.activation_function
        self.add_bias = config.add_bias
        residual_dropout = config.resid_pdrop
        self.is_glu_activation = is_glu(activation_function)

        init_method = InitMethod(config.init_method)
        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size if self.is_glu_activation else intermediate_size,
            bias=self.add_bias,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=self.add_bias, std=std)

        self.dropout = nn.Identity() if residual_dropout == 0 else Dropout_TP(residual_dropout)

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        # GLU is a special case and needs to be handled explicitely
        if self.is_glu_activation:
            weight = safetensors_weight_manager.get_slice(prefix + "c_fc.weight")

            shape = weight.get_shape()
            stride = divide_if_divisible(
                shape[0],
                tp_world_size * 2,
                f"split dimension ({0}) is not divisible by 2 x tensor parallel world size (2 x {tp_world_size})",
            )

            # split weight tensors into gate and non-gate
            start_end = (tp_rank * stride, (tp_rank + 1) * stride)
            weight_1 = tensor_parallel_split_safetensor_slice(weight, 0, start_end)
            if self.add_bias:
                bias = safetensors_weight_manager.get_slice(prefix + "c_fc.bias")
                bias_1 = tensor_parallel_split_safetensor_slice(bias, 0, start_end)

            start_end = (
                (tp_world_size + tp_rank) * stride,
                (tp_world_size + tp_rank + 1) * stride,
            )
            weight_2 = tensor_parallel_split_safetensor_slice(weight, 0, start_end)
            if self.add_bias:
                bias_2 = tensor_parallel_split_safetensor_slice(bias, 0, start_end)

            state_dict = {"weight": torch.cat([weight_1, weight_2])}
            if self.add_bias:
                state_dict["bias"] = torch.cat([bias_1, bias_2])

            self.c_fc.load_state_dict(state_dict)
        else:
            self.c_fc.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix=prefix + "c_fc.")

        self.c_proj.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix=prefix + "c_proj.")
