import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Shard

from ....utils import ProcessGroupManager
from ...modeling_utils_TP import DTensorModule


class DesyncResidualRMSNorm_TP(nn.RMSNorm, DTensorModule):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__(normalized_shape, eps=eps)

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Shard(0)]
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype

        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)

        return self.weight.to_local() * input.to(input_dtype)


_NORMALIZATION_FUNCTIONS = {"rmsnorm": DesyncResidualRMSNorm_TP}


def get_desync_residual_normalization_function_TP(name: str, normalized_shape: int, eps: float = 1e-5) -> nn.LayerNorm:
    if name in _NORMALIZATION_FUNCTIONS:
        return _NORMALIZATION_FUNCTIONS[name](normalized_shape, eps=eps)

    raise ValueError(f"unexpected `normalization_function` {name}")
