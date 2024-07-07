import torch
import torch.nn.functional as F

from ...modeling_utils_TP import RowParallelLinear
from .TP import ensemble_reduce_from_tensor_parallel_region


class EnsembleRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = F.linear(input, self.weight, None)
        input = ensemble_reduce_from_tensor_parallel_region(input)
        if self.bias is not None:
            input = input + self.bias / self.tp_world_size
        return input
