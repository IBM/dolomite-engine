import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Partial, Shard

from ....distributed import dtensor_to_tensor, tensor_to_dtensor
from ....utils import ProcessGroupManager
from ...modeling_utils import ParameterizedLinear
from ...modeling_utils_TP import DTensorModule, RowParallelLinear


class DesyncResidualLinear_TP(ParameterizedLinear, DTensorModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, std)

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Shard(0)]
            )
        )
        if bias:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Shard(0)]
                )
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, dtensor_to_tensor(self.weight), dtensor_to_tensor(self.bias))


class DesyncResidualRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=Shard(-1))
        input = F.linear(input, self.weight, self.bias)
        input = input + tensor_to_dtensor(
            residual / ProcessGroupManager.get_tensor_parallel_world_size(),
            device_mesh=self.tp_mesh,
            current_placement=Partial(),
        )
        input = dtensor_to_tensor(input, device_mesh=self.tp_mesh, desired_placement=self.output_placement)
        return input
