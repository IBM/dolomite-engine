import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard

from .....utils import ProcessGroupManager
from ...TP import dtensor_to_tensor, tensor_to_dtensor


class LayerNorm_TP(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__(normalized_shape, eps=eps)

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

        if sequence_parallel:
            if use_padding_free_transformer:
                self.placement = Shard(0)
            else:
                self.placement = Shard(1)
        else:
            self.placement = Replicate()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, current_placement=self.placement)
        input = super().forward(input)
        input = dtensor_to_tensor(input, desired_placement=self.placement)
        return input
