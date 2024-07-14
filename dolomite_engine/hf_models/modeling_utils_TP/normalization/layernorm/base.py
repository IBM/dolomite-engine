import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard

from .....utils import ProcessGroupManager
from ...TP import dtensor_to_tensor, tensor_to_dtensor


class LayerNorm_TP(nn.LayerNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, sequence_parallel: bool = False) -> None:
        super().__init__(normalized_shape, eps=eps)

        self.sequence_parallel = sequence_parallel

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )
        self.bias = nn.Parameter(
            DTensor.from_local(
                self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, current_placement=Shard(1) if self.sequence_parallel else Replicate())
        input = super().forward(input)
        input = dtensor_to_tensor(input, desired_placement=Shard(1) if self.sequence_parallel else Replicate())
        return input
