import math

import torch.nn as nn

from ...config import CommonConfig
from ...enums import InitMethod
from ...modeling_utils import MLP, get_activation_function, is_glu
from ..dropout import Dropout_TP
from ..linear import ColumnParallelLinear, RowParallelLinear


class MLP_TP(MLP):
    def __init__(
        self, config: CommonConfig, use_padding_free_transformer: bool = False, sequence_parallel: bool = False
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        intermediate_size = config.n_inner
        activation_function = config.activation_function
        self.add_bias = config.add_bias
        residual_dropout = config.resid_pdrop
        self.is_glu_activation = is_glu(activation_function)

        init_method = InitMethod(config.init_method)
        initializer_range = config.initializer_range
        m_width = config.m_width
        num_layers = config.num_layers

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size if self.is_glu_activation else intermediate_size,
            bias=self.add_bias,
            std=std,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=self.add_bias,
            std=std,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.dropout = (
            nn.Identity()
            if residual_dropout == 0
            else Dropout_TP(
                residual_dropout,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )
        )
