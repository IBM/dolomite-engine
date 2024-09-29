import torch
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Shard

from ....utils import is_dtensors_enabled
from ...modeling_utils_TP import RowParallelLinear, tensor_to_dtensor


class ParallelRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor) -> DTensor:
        if is_dtensors_enabled():
            input = tensor_to_dtensor(input, current_placement=Shard(-1))
            input = F.linear(input, self.weight, self.bias)
        else:
            assert self.bias is None

            input = F.linear(input, self.weight.to_local(), None)

        return input
