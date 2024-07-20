import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Placement

from .TP import dtensor_to_tensor, tensor_to_dtensor


def tensor_to_dtensor_hook(
    module: nn.Module,
    inputs: tuple[torch.Tensor],
    current_placement: Placement,
    desired_placement: Placement | None = None,
) -> tuple[DTensor]:
    assert len(inputs) == 1
    input = inputs[0]

    input = tensor_to_dtensor(input, current_placement=current_placement, desired_placement=desired_placement)

    return (input,)


def dtensor_to_tensor_hook(
    module: nn.Module,
    inputs: tuple[DTensor],
    output: DTensor,
    desired_placement: Placement | None = None,
    grad_placement: Placement | None = None,
) -> torch.Tensor:
    return dtensor_to_tensor(output, desired_placement=desired_placement, grad_placement=grad_placement)
