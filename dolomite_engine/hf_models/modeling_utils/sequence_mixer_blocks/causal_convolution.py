# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ....utils import divide_if_divisible
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import get_activation_function, is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..mlp_blocks.mlp import _get_std_for_linear


class CausalConvolution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int,
        activation_function: str,
        add_bias: bool,
        initializer_range: float | None,
        m_width: float,
        init_method: str,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> None:
        if use_padding_free_transformer:
            raise NotImplementedError()

        self.layer_idx = layer_idx

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.input_projection = ParameterizedLinear(hidden_size, in_channels, bias=add_bias)

        divide_if_divisible(in_channels, num_groups)
        divide_if_divisible(out_channels, num_groups)

        if is_glu(activation_function):
            intermediate_size = divide_if_divisible(out_channels, 2)
        else:
            intermediate_size = out_channels

        self.conv1d = ParameterizedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=add_bias,
            padding=kernel_size - 1,
            std=std,
        )

        self.activation_function = get_activation_function(activation_function)

        self.output_projection = ParameterizedLinear(intermediate_size, hidden_size, bias=add_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)
        if input_state is not None:
            input_state = input_state.roll(shifts=-1, dims=-1)

        hidden_states = self.input_projection(hidden_states)

        hidden_states = self.conv1d(hidden_states)
        hidden_states = self.activation_function(hidden_states)

        hidden_states = self.output_projection(hidden_states)

        return hidden_states
