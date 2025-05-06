import math

import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_cute_kernels_available
from ...cache import GenerationCache
from ..linear import ParameterizedLinear
from .padding import compute_cu_seqlens_from_attention_mask, pack_sequence, unpack_sequence


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
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer

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
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert cache_params is None
            assert attention_mask is None
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            batch_size, sequence_length = input.size()[:2]

            if attention_mask is not None:
                cu_seqlens = compute_cu_seqlens_from_attention_mask(attention_mask)
                input = pack_sequence(input=input, cu_seqlens=cu_seqlens)

        input = self.input_projection(input)
        input = input.view(*input.size()[:-1], self.num_heads, self.state_head_dim)

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
            cache_params.update(state=input[:, -1, ...], num_tokens_added=input.size(1), layer_idx=self.layer_idx)

        if not self.use_padding_free_transformer and attention_mask is not None:
            input = unpack_sequence(
                input=input, cu_seqlens=cu_seqlens, desired_shape=(batch_size, sequence_length, *input.size()[1:])
            )

        input = input.view(*input.size()[:-2], -1)
        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)
