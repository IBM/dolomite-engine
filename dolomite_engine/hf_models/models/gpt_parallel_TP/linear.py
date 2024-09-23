import torch
from torch.distributed._tensor.placement_types import Shard

from ...modeling_utils_TP import RowParallelLinear, tensor_to_dtensor


class ParallelRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, current_placement=Shard(-1))
        input = super().forward(input)
        return input
