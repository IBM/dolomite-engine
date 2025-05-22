# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_cute_kernels_available
from ...parameter import mark_parameter_as_mup_learning_rate
from ..activations import get_activation_function, is_glu
from ..activations.glu import GLUActivation
from ..linear import ParameterizedLinear


if is_cute_kernels_available():
    from cute_kernels import fused_swiglu_cute


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
    ) -> None:
        super().__init__()

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.c_fc = ParameterizedLinear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            bias=add_bias,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        self.c_proj = ParameterizedLinear(
            intermediate_size, hidden_size, bias=add_bias, std=std / math.sqrt(2 * num_layers)
        )

        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

        mark_parameter_as_mup_learning_rate(self.c_fc.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.act, GLUActivation) and is_kernel_allowed(Kernel.fused_swiglu_cute):
            up_weight, gate_weight = self.c_fc.weight.chunk(2, dim=0)

            hidden_states = fused_swiglu_cute(
                x=hidden_states,
                gate_weight=gate_weight,
                up_weight=up_weight,
                down_weight=self.c_proj.weight,
                memory_efficient=True,
            )
        else:
            hidden_states = self.c_fc(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.c_proj(hidden_states)

        hidden_states = self.dropout(hidden_states)

        return hidden_states


def _get_std_for_linear(initializer_range: float, init_method: str, m_width: float | None) -> float:
    std = initializer_range
    if init_method == "mup":
        std /= math.sqrt(m_width)
    elif init_method != "normal":
        raise ValueError(f"unexpected init_method ({init_method})")

    return std


def interleave_up_gate_tensor_for_mlp(up_weight: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    return torch.cat([up_weight, gate_weight])


def split_up_gate_tensor_for_mlp(c_fc_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return c_fc_weight.chunk(2)
