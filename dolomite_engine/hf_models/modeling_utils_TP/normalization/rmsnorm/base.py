import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from .....utils import ProcessGroupManager
from ...TP import dtensor_to_tensor, get_module_placements, tensor_to_dtensor


class RMSNorm_TP(nn.RMSNorm):
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

        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, current_placement=self.placement)
        input = super().forward(input)
        input = dtensor_to_tensor(input, desired_placement=self.placement)
        return input
