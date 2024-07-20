import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Placement

from ...utils import ProcessGroupManager
from ..utils import divide_if_divisible


def tensor_parallel_split_safetensor_slice(slice, dim: int, start_end: tuple[int, int] | None = None) -> torch.Tensor:
    shape = slice.get_shape()
    dimensionality = len(shape)
    assert dimensionality in [1, 2], f"tensor should be either 1 or 2 dimensional but {dimensionality} was found"

    tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

    if start_end is None:
        stride = divide_if_divisible(
            shape[dim],
            tp_world_size,
            f"split dimension ({dim}) is not divisible by tensor parallel world size ({tp_world_size})",
        )
        start_index = tp_rank * stride
        end_index = (tp_rank + 1) * stride
    else:
        start_index = start_end[0]
        end_index = start_end[1]

    if dimensionality == 1:
        # bias tensor
        assert dim == 0, f"dim has to 0 for a bias tensor but dim ({dim}) was passed"
        return slice[start_index:end_index]
    elif dimensionality == 2:
        assert 0 <= dim <= 1, f"dim has to 0 or 1 for a weight tensor but dim ({dim}) was passed"
        # weight tensor
        if dim == 0:
            return slice[start_index:end_index, :]
        else:
            return slice[:, start_index:end_index]
    else:
        raise RuntimeError("this code should not be reachable")


def tensor_to_dtensor(
    tensor: torch.Tensor, current_placement: Placement, desired_placement: Placement | None = None
) -> DTensor:
    tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

    dtensor = DTensor.from_local(tensor, device_mesh=tp_mesh, run_check=False, placements=[current_placement])
    if desired_placement is not None:
        dtensor = dtensor.redistribute(device_mesh=tp_mesh, placements=[desired_placement])

    return dtensor


def dtensor_to_tensor(
    dtensor: DTensor, desired_placement: Placement | None = None, grad_placement: Placement | None = None
) -> torch.Tensor:
    if desired_placement is not None:
        dtensor = dtensor.redistribute(
            device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[desired_placement]
        )

    tensor = dtensor.to_local(grad_placements=None if grad_placement is None else [grad_placement])

    return tensor


@torch.no_grad()
def modify_state_dict_to_dtensor_dict(module: nn.Module, state_dict: dict) -> dict:
    result = {}
    for key, tensor in state_dict.items():
        device_mesh = getattr(module, key).device_mesh
        placements = getattr(module, key).placements
        result[key] = DTensor.from_local(tensor, device_mesh=device_mesh, placements=placements)
    return result
