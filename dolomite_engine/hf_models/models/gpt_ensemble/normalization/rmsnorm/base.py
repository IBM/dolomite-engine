import numbers

import torch
import torch.nn as nn


class EnsembleRMSNorm(nn.RMSNorm):
    def __init__(self, normalized_shape: int, tp_world_size: int, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)

        self.tp_world_size = tp_world_size

        self.weight = nn.Parameter(torch.ones(tp_world_size * normalized_shape))
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

        weight = self.weight.view(self.tp_world_size, self.normalized_shape[0]).unsqueeze(1).unsqueeze(1)
        input = weight * input

        return input
