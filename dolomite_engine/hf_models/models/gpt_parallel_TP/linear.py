import torch
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Shard

from ...modeling_utils_TP import RowParallelLinear, tensor_to_dtensor


class ParallelRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor) -> DTensor:
        input = tensor_to_dtensor(input, current_placement=Shard(-1))
        input = F.linear(input, self.weight, self.bias)
        return input
