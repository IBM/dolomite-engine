import torch
import torch.nn as nn

from ...utils import get_cuda_rng_tracker


class Dropout_TP(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            with get_cuda_rng_tracker().fork():
                input = super().forward(input)

        return input
