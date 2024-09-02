import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Partial, Shard

from ...modeling_utils_TP import RowParallelLinear, dtensor_to_tensor, tensor_to_dtensor


class EnsembleRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, current_placement=Shard(-1))
        residual = tensor_to_dtensor(residual / self.tp_world_size, current_placement=Partial())

        input = F.linear(input, self.weight, self.bias)
        input = input + residual

        input = dtensor_to_tensor(input, desired_placement=self.output_placement)

        return input
