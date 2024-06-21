import numbers

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()

        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))

        self.eps = eps

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype

        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)

        input = input.to(input_dtype)
        if self.elementwise_affine:
            input = self.weight * input

        return input

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
