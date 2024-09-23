import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Shard

from ...modeling_utils_TP import ColumnParallelLinear, dtensor_to_tensor


class LadderColumnParallelLinear(ColumnParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = F.linear(input, self.weight, self.bias)
        input = dtensor_to_tensor(input, desired_placement=Shard(-1))
        return input
