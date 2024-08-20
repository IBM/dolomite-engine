import math

import torch.nn as nn

from ....utils import ProcessGroupManager
from ...enums import InitMethod
from ...modeling_utils import ParameterizedLinear, get_activation_function, is_glu
from ...modeling_utils_TP import ColumnParallelLinear, Dropout_TP, RowParallelLinear
from ...utils import divide_if_divisible
from ..gpt_dolomite_TP.mlp import MLP_TP
from ..gpt_ensemble import GPTEnsembleConfig


class EnsembleMLP_TP(MLP_TP):
    def __init__(self, config: GPTEnsembleConfig, layer_idx: int = None) -> None:
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

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        if config.reduce_pattern[layer_idx]["attention"]:
            self.c_fc = ColumnParallelLinear(
                hidden_size,
                2 * intermediate_size if self.is_glu_activation else intermediate_size,
                bias=self.add_bias,
                std=std,
            )
        else:
            self.c_fc = ParameterizedLinear(
                hidden_size,
                divide_if_divisible(
                    2 * intermediate_size if self.is_glu_activation else intermediate_size,
                    tp_world_size,
                    "",
                ),
                bias=self.add_bias,
                std=std,
            )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        if layer_idx == config.n_layer - 1 or config.reduce_pattern[layer_idx]["mlp"]:
            self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=self.add_bias, std=std)
        else:
            self.c_proj = ParameterizedLinear(
                divide_if_divisible(intermediate_size, tp_world_size, ""), hidden_size, bias=self.add_bias, std=std
            )

        self.dropout = nn.Identity() if residual_dropout == 0 else Dropout_TP(residual_dropout)
