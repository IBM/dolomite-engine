import math

import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_cute_kernels_available
from ...cache import GenerationCache
from ..linear import ParameterizedLinear


if is_cute_kernels_available():
    from cute_kernels import rnn_cute, rnn_torch


class RNN(nn.Module):
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
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx

        self.input_head_dim = divide_if_divisible(self.input_size, self.num_heads, "")
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.input_projection = ParameterizedLinear(self.input_size, self.state_size, bias=add_bias, std=std)
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.output_projection = ParameterizedLinear(self.state_size, self.output_size, bias=False, std=std)

        self.factor = 1 / math.sqrt(self.input_size + self.state_head_dim)
        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        cache_params: GenerationCache | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = input.size()

        input = self.input_projection(input)
        input = input.view(batch_size, sequence_length, self.num_heads, -1)

        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)

        input = input * self.factor
        weight = self.state_weight * self.factor

        if is_kernel_allowed(Kernel.rnn_cute):
            input = rnn_cute(
                input=input,
                weight=weight,
                input_state=input_state,
                gradient_clipping=self.gradient_clipping,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            input = rnn_torch(
                input=input,
                weight=weight,
                input_state=input_state,
                gradient_clipping=self.gradient_clipping,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        if cache_params is not None:
            cache_params.update(state=input[:, -1, ...], num_tokens_added=sequence_length, layer_idx=self.layer_idx)

        input = input.view(batch_size, sequence_length, -1)
        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)
