import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Replicate, Shard

from .TP import dtensor_to_tensor, get_module_placements, tensor_to_dtensor


class Dropout_TP(nn.Dropout):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__(p, inplace)
        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            input = tensor_to_dtensor(input, current_placement=self.placement)
            input = super().forward(input)
            input = dtensor_to_tensor(input, desired_placement=self.placement)

        return input
