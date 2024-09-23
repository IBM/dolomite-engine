import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard

from ......utils import ProcessGroupManager
from .....modeling_utils_TP import DTensorModule, dtensor_to_tensor, get_module_placements, tensor_to_dtensor


class ParallelRMSNorm_TP(nn.RMSNorm, DTensorModule):
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
                2, self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )

        self.input_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        if sequence_parallel:
            if use_padding_free_transformer:
                self.output_placement = Shard(1)
            else:
                self.output_placement = Shard(2)
        else:
            self.output_placement = Replicate()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        input = input.float()

        input = tensor_to_dtensor(input, current_placement=self.input_placement)

        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)
        input = self.weight * input.to(input_dtype).unsqueeze(0)

        input = dtensor_to_tensor(input, desired_placement=self.output_placement)

        return input
