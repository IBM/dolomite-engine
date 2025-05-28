# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_cute_kernels_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import get_activation_function, is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from .causal_convolution import causal_convolution
from .packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence
from .rnn import RNN


if is_cute_kernels_available():
    from cute_kernels import gru_cute, gru_torch


class GroupedLinear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        std: float | None = None,
    ) -> None:
        self.std = std
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.in_dim = divide_if_divisible(in_channels, groups, "in_channels must be divisible by groups")
        self.out_dim = divide_if_divisible(out_channels, groups, "out_channels must be divisible by groups")
        self.weight = nn.Parameter(torch.empty(self.groups, self.in_dim, self.out_dim))
        self.reset_parameters()

        # mark_parameter_as_no_weight_decay(self.bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.zero_()

    def forward(self, x):
        x_size = x.size()
        # X: ..., groups * in_dim
        x = x.view(-1, self.groups, self.in_dim)
        # X: ..., groups, in_dim
        x = x.transpose(1, 0)
        # X: groups,  *, in_dim
        # weight: groups, in_dim, out_dim
        y = torch.bmm(x, self.weight)
        # y: groups, *, out_dim
        y = y.transpose(1, 0)
        # y: *, groups, out_dim
        y = y.reshape(*(x_size[:-1]), self.out_channels)
        return y


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
        head_activation: str = "tanh",
        factor: float = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.input_projection = ParameterizedLinear(self.input_size, self.input_size, bias=add_bias, std=std)

        self.head_activation = get_activation_function(head_activation)
        # self.head_projection = ParameterizedConv1d(
        #     in_channels=self.input_size,
        #     out_channels=3 * self.state_size,
        #     kernel_size=1,
        #     groups=self.num_heads,
        #     bias=add_bias,
        #     std=std,
        # )
        self.head_projection = GroupedLinear(
            in_channels=self.input_size,
            out_channels=3 * self.state_size,
            groups=self.num_heads,
            std=std,
        )

        self.state_weight = nn.Parameter(torch.empty(3 * self.num_heads, self.state_head_dim, self.state_head_dim))

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        # self.output_head_projection = ParameterizedConv1d(
        #     in_channels=self.state_size,
        #     out_channels=self.input_size,
        #     kernel_size=1,
        #     groups=self.num_heads,
        #     bias=add_bias,
        #     std=std,
        # )

        self.output_head_projection = GroupedLinear(
            in_channels=self.state_size, out_channels=self.input_size, groups=self.num_heads, std=std
        )
        self.ln_output_head = nn.GroupNorm(num_groups=self.num_heads, num_channels=self.input_size)

        self.output_projection = ParameterizedLinear(self.state_size, self.output_size, bias=False, std=std)

        if factor is None:
            factor = 1 / math.sqrt(2 * self.state_head_dim)

        self.factor = factor

        self.reset_parameters()

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        # mark_parameter_as_no_weight_decay(self.state_weight)

    def forward(
        self,
        input: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert cache_params is None
            assert attention_mask is None
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            batch_size, sequence_length = input.size()[:2]

            if attention_mask is not None:
                cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
                input = pack_sequence(inputs=input, cu_seqlens=cu_seqlens)

        input = self.input_projection(input)
        input = self.head_activation(input)
        input = self.head_projection(input)

        input = input * self.factor
        weight = self.state_weight * self.factor

        input, forget_input, reset_input = input.chunk(3, dim=-1)
        weight, forget_weight, reset_weight = weight.chunk(3, dim=0)

        input, forget_input, reset_input = [
            i.view(*input.size()[:-1], self.num_heads, self.state_head_dim) for i in (input, forget_input, reset_input)
        ]

        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)

        input = (gru_cute if is_kernel_allowed(Kernel.gru_cute) else gru_torch)(
            input=input,
            weight=weight,
            forget_input=forget_input,
            forget_weight=forget_weight,
            reset_input=reset_input,
            reset_weight=reset_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        input = self.output_head_projection(input.flatten(-2, -1))
        input_size = input.size()
        input = self.ln_output_head(input.flatten(0, 1)).view(input_size)

        if not self.use_padding_free_transformer and attention_mask is not None:
            input = unpack_sequence(
                inputs=input, cu_seqlens=cu_seqlens, desired_shape=(batch_size, sequence_length, *input.size()[1:])
            )

        if cache_params is not None:
            input_state = input[:, -1].view(input.size(0), -1)
            cache_params.update(state=input_state, num_tokens_added=input.size(1), layer_idx=self.layer_idx)

        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping},\nweight_shape= {str(self.state_weight.shape)},\nfactor={self.factor}"
