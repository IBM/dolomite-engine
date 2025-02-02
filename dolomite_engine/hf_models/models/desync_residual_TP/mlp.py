import math

import torch
import torch.nn as nn

from ....utils import ProcessGroupManager, divide_if_divisible
from ...enums import InitMethod
from ...modeling_utils import get_activation_function, is_glu
from ...modeling_utils_TP import ColumnParallelLinear, Dropout_TP
from ..desync_residual import DesyncResidualConfig
from ..gpt_dolomite_TP.mlp import MLP_TP
from .linear import DesyncResidualLinear_TP, DesyncResidualRowParallelLinear


class DesyncResidualMLP_TP(MLP_TP):
    def __init__(self, config: DesyncResidualConfig, layer_idx: int = None) -> None:
        nn.Module.__init__(self)

        hidden_size = config.n_embd
        intermediate_size = config.n_inner
        activation_function = config.activation_function
        self.add_bias = config.add_bias
        residual_dropout = config.resid_pdrop
        self.is_glu_activation = is_glu(activation_function)
        self.m_residual = config.m_residual

        init_method = InitMethod(config.init_method)
        initializer_range = config.initializer_range
        m_width = config.m_width
        self.n_layer = config.n_layer
        self.layer_idx = layer_idx

        self.current_mlp_all_reduce = layer_idx == config.n_layer - 1 or config.reduce_pattern[layer_idx]["mlp"]
        self.current_attention_all_reduce = config.reduce_pattern[layer_idx]["attention"]

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        if self.current_attention_all_reduce:
            self.c_fc = ColumnParallelLinear(
                hidden_size,
                2 * intermediate_size if self.is_glu_activation else intermediate_size,
                bias=self.add_bias,
                std=std,
            )
        else:
            self.c_fc = DesyncResidualLinear_TP(
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

        std = initializer_range / math.sqrt(2 * self.n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        if self.current_mlp_all_reduce:
            self.c_proj = DesyncResidualRowParallelLinear(intermediate_size, hidden_size, bias=self.add_bias, std=std)
        else:
            self.c_proj = DesyncResidualLinear_TP(
                divide_if_divisible(intermediate_size, tp_world_size, ""), hidden_size, bias=self.add_bias, std=std
            )

        assert residual_dropout == 0, "residual dropout is not supported with DesyncResidual"
        self.dropout = nn.Identity() if residual_dropout == 0 else Dropout_TP(residual_dropout)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        if self.current_mlp_all_reduce:
            hidden_states = self.c_proj(hidden_states, residual)
        else:
            hidden_states = self.c_proj(hidden_states)
            hidden_states = hidden_states + residual

        hidden_states = self.dropout(hidden_states)
        return hidden_states
