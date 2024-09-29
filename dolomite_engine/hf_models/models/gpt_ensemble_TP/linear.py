import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ReduceOp
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Partial, Shard

from ....utils import ProcessGroupManager, is_dtensors_enabled
from ...modeling_utils import ParameterizedLinear
from ...modeling_utils_TP import (
    RowParallelLinear,
    dtensor_to_tensor,
    reduce_from_tensor_parallel_region,
    tensor_to_dtensor,
)


class EnsembleLinear_TP(ParameterizedLinear):
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
        return F.linear(input, self.weight.to_local(), None if self.bias is None else self.bias.to_local())


class EnsembleRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if is_dtensors_enabled():
            input = tensor_to_dtensor(input, current_placement=Shard(-1))
            residual = tensor_to_dtensor(residual / self.tp_world_size, current_placement=Partial())

            input = F.linear(input, self.weight, self.bias)
            input = input + residual

            input = dtensor_to_tensor(input, desired_placement=self.output_placement)
        else:
            assert not self.use_padding_free_transformer
            assert not self.sequence_parallel

            input = F.linear(input, self.weight.to_local(), None)
            input = input + residual / self.tp_world_size
            input = reduce_from_tensor_parallel_region(input)

            if self.bias is not None:
                input = input + self.bias.to_local()

        return input
