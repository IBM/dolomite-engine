import torch
import torch.nn as nn


class RMSNorm(nn.RMSNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype

        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)

        return self.weight * input.to(input_dtype)
