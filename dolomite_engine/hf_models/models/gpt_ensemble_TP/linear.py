import torch

from ...modeling_utils_TP import RowParallelLinear


class EnsembleRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = super().forward(input)
        input = input / self.tp_world_size
        return input
