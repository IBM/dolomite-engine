import torch
import torch.distributed
import torch.nn.functional as F

from ....distributed import dtensor_to_tensor
from ...modeling_utils_TP import ColumnParallelLinear, RowParallelLinear


class LadderColumnParallelLinear(ColumnParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, dtensor_to_tensor(self.weight), dtensor_to_tensor(self.bias))


class LadderRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, dtensor_to_tensor(self.weight), dtensor_to_tensor(self.bias))
