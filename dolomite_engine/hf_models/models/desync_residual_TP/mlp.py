import math

import torch
import torch.nn as nn

from ....utils import ProcessGroupManager, divide_if_divisible
from ...enums import InitMethod
from ...modeling_utils import get_activation_function, is_glu
from ...modeling_utils.mlp_block.mlp import _get_std_for_linear
from ...modeling_utils_TP import MLP_TP, ColumnParallelLinear
from .linear import DesyncResidualLinear_TP, DesyncResidualRowParallelLinear


class DesyncResidualMLP_TP(MLP_TP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: InitMethod,
        initializer_range: float,
        m_width: float,
        m_residual: float,
        num_layers: int,
        pretraining_tensor_parallel_size: int,
        all_reduce: bool,
        attention_did_all_reduce: bool,
    ) -> None:
        nn.Module.__init__(self)

        self.m_residual = m_residual
        self.tp_world_size = pretraining_tensor_parallel_size
        self.num_layers = num_layers
        self.all_reduce = all_reduce

        is_glu_activation = is_glu(activation_function)

        self.all_reduce = all_reduce
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        if attention_did_all_reduce:
            self.c_fc = ColumnParallelLinear(
                hidden_size,
                2 * intermediate_size if is_glu_activation else intermediate_size,
                bias=add_bias,
                std=std,
            )
        else:
            self.c_fc = DesyncResidualLinear_TP(
                hidden_size,
                divide_if_divisible(
                    2 * intermediate_size if is_glu_activation else intermediate_size,
                    tp_world_size,
                    "",
                ),
                bias=add_bias,
                std=std,
            )

        self.act = get_activation_function(activation_function)

        std /= math.sqrt(2 * self.num_layers)

        if self.all_reduce:
            self.c_proj = DesyncResidualRowParallelLinear(intermediate_size, hidden_size, bias=add_bias, std=std)
        else:
            self.c_proj = DesyncResidualLinear_TP(
                divide_if_divisible(intermediate_size, tp_world_size, ""), hidden_size, bias=add_bias, std=std
            )

        assert dropout == 0, "residual dropout is not supported with DesyncResidual"

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        if self.all_reduce:
            hidden_states = self.c_proj(hidden_states, residual)
        else:
            hidden_states = self.c_proj(hidden_states)
            hidden_states = hidden_states + residual

        return hidden_states
