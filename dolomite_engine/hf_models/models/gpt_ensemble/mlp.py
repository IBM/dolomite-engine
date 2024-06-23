import math

import torch.nn as nn

from ...enums import InitMethod
from ...modeling_utils import get_activation_function, is_glu
from ...utils import divide_if_divisible
from ..gpt_dolomite.mlp import MLP
from .config import GPTEnsembleConfig
from .linear import EnsembleLinear


class EnsembleMLP(MLP):
    def __init__(self, config: GPTEnsembleConfig) -> None:
        nn.Module.__init__(self)

        hidden_size = config.n_embd
        intermediate_size = divide_if_divisible(config.n_inner, config.pretraining_tensor_parallel_size, "")
        activation_function = config.activation_function
        add_bias = config.add_bias
        residual_dropout = config.resid_pdrop

        init_method = InitMethod(config.init_method)
        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer

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

        std = initializer_range / math.sqrt(2 * n_layer)
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
