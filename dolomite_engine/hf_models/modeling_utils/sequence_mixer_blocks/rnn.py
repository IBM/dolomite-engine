import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_cute_kernels_available
from ..linear import ParameterizedLinear


if is_cute_kernels_available():
    from cute_kernels import rnn_cute


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        m_width: float,
        num_layers: int,
        add_bias: bool = True,
        initializer_range: float = 1,
        init_method: str = "normal",
        gradient_clipping: float | None = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping

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

        self.reset_parameters()

    def forward(self, input: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length, _ = input.size()

        input = self.input_projection(input)
        input = input.view(batch_size, sequence_length, self.num_heads, -1)

        if is_kernel_allowed(Kernel.rnn_cute):
            input = rnn_cute(
                input=input,
                weight=self.state_weight,
                input_state=input_state,
                gradient_clipping=self.gradient_clipping,
            )
        else:
            input = self._torch_forward(input, input_state)

        input = input.view(batch_size, sequence_length, -1)
        input = self.output_projection(input)

        return input

    def _torch_forward(self, input: torch.Tensor, input_state: torch.Tensor | None = None) -> torch.Tensor:
        if self.gradient_clipping is not None:
            raise NotImplementedError("rnn_torch doesn't support gradient_clipping")

        B, S, N, H = input.size()
        output = torch.empty_like(input)

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

        weight = self.state_weight.unsqueeze(0)
        input = input.unsqueeze(-2)

        # input -> (B, S, N, 1, H)
        # weight -> (1, N, H, H)
        # input_state -> (B, N, H)

        for s in range(S):
            input_state = input_state.unsqueeze(-2)

            # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
            input_state = input_state @ weight + input[:, s, ...]

            input_state = input_state.float()
            input_state = F.tanh(input_state)
            input_state = input_state.type_as(input)

            input_state = input_state.squeeze(-2)

            output[:, s, ...] = input_state

        return output

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)
