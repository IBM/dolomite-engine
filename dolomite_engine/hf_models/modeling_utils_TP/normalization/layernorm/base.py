import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from .....utils import ProcessGroupManager


class LayerNorm_TP(nn.LayerNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight.to_local(), self.bias.to_local(), self.eps)
