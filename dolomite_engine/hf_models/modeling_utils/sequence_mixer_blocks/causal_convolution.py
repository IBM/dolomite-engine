# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_causal_conv1d_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import get_activation_function, is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..mlp_blocks.mlp import _get_std_for_linear


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn


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
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__()

        if use_padding_free_transformer:
            raise NotImplementedError()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.layer_idx = layer_idx
        self.activation_string = activation_function

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.input_projection = ParameterizedLinear(hidden_size, in_channels, bias=add_bias, std=std)

        divide_if_divisible(in_channels, num_groups, "")
        divide_if_divisible(out_channels, num_groups, "")

        if is_glu(self.activation_string):
            intermediate_size = divide_if_divisible(out_channels, 2, "")
        else:
            intermediate_size = out_channels

        self.conv1d = ParameterizedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=add_bias,
            padding=kernel_size - 1,
            groups=num_groups,
            std=std,
        )

        self.activation_function = get_activation_function(self.activation_string)

        self.output_projection = ParameterizedLinear(
            intermediate_size, hidden_size, bias=add_bias, std=std / math.sqrt(2 * num_layers)
        )

        self.casual_conv1d_compatible = (
            self.num_groups == self.in_channels == self.out_channels
            and self.activation_string in [None, "silu", "swish"]
        )

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
        hidden_states = hidden_states.transpose(-1, -2)

        if is_kernel_allowed(Kernel.causal_conv1d) and self.casual_conv1d_compatible:
            hidden_states = causal_conv1d_fn(
                x=hidden_states,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation_string,
            )

            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = self.conv1d(hidden_states)
            hidden_states = hidden_states[..., : -(self.kernel_size - 1)]

            hidden_states = hidden_states.transpose(-1, -2)
            hidden_states = self.activation_function(hidden_states)

        hidden_states = self.output_projection(hidden_states)

        return hidden_states
