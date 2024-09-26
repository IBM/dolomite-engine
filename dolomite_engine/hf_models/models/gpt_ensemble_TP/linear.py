import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed import ReduceOp
from torch.distributed._tensor.placement_types import Partial, Shard

from ....utils import is_dtensors_enabled
from ...modeling_utils_TP import (
    RowParallelLinear,
    dtensor_to_tensor,
    reduce_from_tensor_parallel_region,
    tensor_to_dtensor,
)


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
            input = reduce_from_tensor_parallel_region(input, op=ReduceOp.SUM)

            if self.bias is not None:
                input = input + self.bias.to_local()

        return input
