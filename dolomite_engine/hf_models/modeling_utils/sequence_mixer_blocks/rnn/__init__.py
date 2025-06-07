# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn

from .....enums import Kernel
from .....kernels import is_kernel_allowed
from .....utils import divide_if_divisible, ensure_contiguous, is_cute_kernels_available
from ....cache import GenerationCache
from ....parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ...linear import ParameterizedLinear
from ...normalization import get_normalization_function
from ..packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence
from .torch_implementation import rnn_torch
from .triton_implementation import (
    diagonal_rnn_backward_triton,
    diagonal_rnn_forward_triton,
    diagonal_rnn_varlen_backward_triton,
    diagonal_rnn_varlen_forward_triton,
    rnn_backward_triton,
    rnn_forward_triton,
    rnn_varlen_backward_triton,
    rnn_varlen_forward_triton,
)


class _RNN_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        assert input.dim() in [3, 4]
        assert weight.dim() == 3

        N, H = input.size()[-2:]
        assert weight.size() == (N, H, H)

        if gradient_clipping is not None and gradient_clipping < 0:
            gradient_clipping = -gradient_clipping

        output = torch.empty_like(input)

        kwargs = {"input": input, "weight": weight, "input_state": input_state, "output": output}

        if cu_seqlens is None:
            assert max_seqlen is None

            if H == 1:
                diagonal_rnn_forward_triton(**kwargs)
            else:
                rnn_forward_triton(**kwargs)
        else:
            assert max_seqlen is not None
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            kwargs["cu_seqlens"] = cu_seqlens
            kwargs["max_seqlen_tensor"] = max_seqlen if is_max_seqlen_tensor else None
            kwargs["max_seqlen"] = None if is_max_seqlen_tensor else max_seqlen

            if H == 1:
                diagonal_rnn_varlen_forward_triton(**kwargs)
            else:
                rnn_varlen_forward_triton(**kwargs)

        ctx.save_for_backward(weight, output, input_state, cu_seqlens, max_seqlen)
        ctx.gradient_clipping = gradient_clipping

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        weight, output, input_state, cu_seqlens, max_seqlen = ctx.saved_tensors
        input_grad = torch.empty_like(output)
        weight_grad = torch.zeros_like(weight, dtype=torch.float32)

        H = weight.size(-1)

        kwargs = {
            "weight": weight,
            "output": output,
            "input_state": input_state,
            "output_grad": output_grad,
            "input_grad": input_grad,
            "weight_grad": weight_grad,
            "gradient_clipping": ctx.gradient_clipping,
        }

        if cu_seqlens is None:
            if H == 1:
                diagonal_rnn_backward_triton(**kwargs)
            else:
                rnn_backward_triton(**kwargs)
        else:
            is_max_seqlen_tensor = isinstance(max_seqlen, torch.Tensor)

            kwargs["cu_seqlens"] = cu_seqlens
            kwargs["max_seqlen_tensor"] = max_seqlen if is_max_seqlen_tensor else None
            kwargs["max_seqlen"] = None if is_max_seqlen_tensor else max_seqlen

            if H == 1:
                diagonal_rnn_varlen_backward_triton(**kwargs)
            else:
                rnn_varlen_backward_triton(**kwargs)

        return input_grad, weight_grad, *[None] * 8


def rnn_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """computes multihead RNN recurrent update over the sequence length: tanh(`input_state` @ `weight` + `input`)

    Args:
        input (torch.Tensor): input tensor of shape (B, S, N, H) where N is the number of heads and H is the head
            dimension. Should have shape (T, N, H) and `cu_seqlens` should be passed.
        weight (torch.Tensor): weight tensor of shape (N, H, H)
        input_state (torch.Tensor | None, optional): starting state of shape (B, N, H), None means starting state
            is 0 tensor. Defaults to None.
        gradient_clipping (float | None, optional): gradient clipping for the state gradient in backward, None
            implies no clipping. Defaults to None.
        cu_seqlens (torch.Tensor | None, optional): cumulative sequence length (must contain 0 as first element). Defaults to None.
        max_seqlen (torch.Tensor | int | None, optional): max sequence length in the batch. Defaults to None.

    Returns:
        torch.Tensor: output tensor of shape (B, S, N, H)
    """

    return _RNN_Cute.apply(input, weight, input_state, gradient_clipping, cu_seqlens, max_seqlen)


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
        normalization_function: str | None,
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
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")
        self.is_gated_normalization = normalization_function == "silu_gated_rmsnorm"

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.input_projection = ParameterizedLinear(
            self.input_size,
            self.state_size + (self.state_size if self.is_gated_normalization else 0),
            bias=add_bias,
            std=std,
        )

        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.output_projection = ParameterizedLinear(self.state_size, self.output_size, bias=False, std=std)

        self.norm = get_normalization_function(normalization_function, self.state_size)

        self.factor = 1 / math.sqrt(self.input_size + self.state_head_dim)
        self.reset_parameters()

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        mark_parameter_as_no_weight_decay(self.state_weight)

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

        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)

        input = self.input_projection(input)

        if self.is_gated_normalization:
            input, gate = input.chunk(2, dim=-1)

        input = input.view(*input.size()[:-1], self.num_heads, self.state_head_dim)

        input = input * self.factor
        weight = self.state_weight * self.factor

        input = (rnn_cute if is_kernel_allowed(Kernel.rnn_cute) else rnn_torch)(
            input=input,
            weight=weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if not self.use_padding_free_transformer and attention_mask is not None:
            input = unpack_sequence(
                inputs=input, cu_seqlens=cu_seqlens, desired_shape=(batch_size, sequence_length, *input.size()[1:])
            )

        if cache_params is not None:
            cache_params.update(state=input[:, -1], num_tokens_added=input.size(1), layer_idx=self.layer_idx)

        input = input.view(*input.size()[:-2], -1)

        if self.is_gated_normalization:
            input = self.norm(input, gate)
        else:
            input = self.norm(input)

        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
