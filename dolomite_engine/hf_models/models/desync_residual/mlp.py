import math

import torch
import torch.nn as nn

from ....utils import divide_if_divisible
from ...enums import InitMethod
from ...modeling_utils import get_activation_function, is_glu
from .config import DesyncResidualConfig
from .linear import DesyncResidualLinear


class DesyncResidualMLP(nn.Module):
    def __init__(self, config: DesyncResidualConfig, layer_idx: int = None) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.m_residual = config.m_residual
        self.tp_world_size = config.pretraining_tensor_parallel_size
        self.n_layer = config.n_layer
        self.current_mlp_all_reduce = layer_idx == self.n_layer - 1 or config.reduce_pattern[layer_idx]["mlp"]

        hidden_size = config.hidden_size
        intermediate_size = divide_if_divisible(config.n_inner, config.pretraining_tensor_parallel_size, "")
        activation_function = config.activation_function
        add_bias = config.add_bias
        residual_dropout = config.resid_pdrop

        init_method = InitMethod(config.init_method)
        initializer_range = config.initializer_range
        m_width = config.m_width

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = DesyncResidualLinear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            tensor_parallel_size=config.pretraining_tensor_parallel_size,
            bias=add_bias,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * self.n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = DesyncResidualLinear(
            intermediate_size,
            hidden_size,
            tensor_parallel_size=config.pretraining_tensor_parallel_size,
            bias=add_bias,
            std=std,
        )

        assert residual_dropout == 0, "residual dropout is not supported with DesyncResidual"
        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        if self.current_mlp_all_reduce:
            hidden_states = hidden_states + residual / self.tp_world_size
            hidden_states = hidden_states.sum(dim=0)
        else:
            hidden_states = hidden_states + residual

        hidden_states = self.dropout(hidden_states)
        return hidden_states
