import math

import torch
import torch.nn as nn

from ...enums import InitMethod
from ...modeling_utils import get_activation_function, is_glu
from ...utils import divide_if_divisible
from ..gpt_dolomite.mlp import MLP
from .config import GPTEnsembleConfig
from .linear import EnsembleLinear


class EnsembleMLP(MLP):
    def __init__(self, config: GPTEnsembleConfig, layer_idx: int = None) -> None:
        nn.Module.__init__(self)

        self.layer_idx = layer_idx
        self.reduce_pattern = config.reduce_pattern

        hidden_size = config.n_embd
        intermediate_size = divide_if_divisible(config.n_inner, config.pretraining_tensor_parallel_size, "")
        activation_function = config.activation_function
        add_bias = config.add_bias
        residual_dropout = config.resid_pdrop

        init_method = InitMethod(config.init_method)
        initializer_range = config.initializer_range
        m_width = config.m_width
        self.n_layer = config.n_layer

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = EnsembleLinear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            tensor_parallel_size=config.pretraining_tensor_parallel_size,
            bias=add_bias,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range * config.pretraining_tensor_parallel_size / math.sqrt(2 * self.n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = EnsembleLinear(
            intermediate_size,
            hidden_size,
            tensor_parallel_size=config.pretraining_tensor_parallel_size,
            bias=add_bias,
            std=std,
        )

        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.reduce_pattern[self.layer_idx]["attention"]:
            assert hidden_states.dim() == 3
            hidden_states = hidden_states.unsqueeze(0)
        else:
            assert hidden_states.dim() == 4

        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        if self.layer_idx == self.n_layer - 1 or self.reduce_pattern[self.layer_idx]["mlp"]:
            hidden_states = hidden_states.sum(dim=0)

        hidden_states = self.dropout(hidden_states)
        return hidden_states
