import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

from ...utils import ProcessGroupManager
from ..utils import divide_if_divisible


def tensor_parallel_split_safetensor_slice(slice, dim: int, start_end: tuple[int, int] | None = None) -> torch.Tensor:
    shape = slice.get_shape()
    dimensionality = len(shape)
    assert dimensionality <= 3, f"tensor should be <= 3 dimensional but {dimensionality} was found"

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

    assert 0 <= dim <= dimensionality - 1, f"dim ({dim}) has to <= dimenstionality ({dimensionality})"

    if dim == 0:
        output = slice[start_index:end_index]
    elif dim == 1:
        output = slice[:, start_index:end_index]
    elif dim == 2:
        output = slice[..., start_index:end_index]

    return output


def tensor_to_dtensor(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    current_placement: Placement | list[Placement],
    desired_placement: Placement | list[Placement] | None = None,
    run_check: bool = False,
) -> DTensor:
    if isinstance(current_placement, Placement):
        current_placement = [current_placement]

    dtensor = DTensor.from_local(tensor, device_mesh=device_mesh, run_check=run_check, placements=current_placement)

    if desired_placement is not None:
        if isinstance(desired_placement, Placement):
            desired_placement = [desired_placement]

        dtensor = dtensor.redistribute(device_mesh=device_mesh, placements=desired_placement)

    return dtensor


def dtensor_to_tensor(
    dtensor: DTensor,
    device_mesh: DeviceMesh | None = None,
    desired_placement: Placement | list[Placement] | None = None,
    grad_placement: Placement | list[Placement] | None = None,
) -> torch.Tensor:
    if desired_placement is not None:
        if isinstance(desired_placement, Placement):
            desired_placement = [desired_placement]

        assert device_mesh is not None

        dtensor = dtensor.redistribute(device_mesh=device_mesh, placements=desired_placement)

    if grad_placement is not None and isinstance(grad_placement, Placement):
        grad_placement = [grad_placement]

    tensor = dtensor.to_local(grad_placements=grad_placement)

    return tensor


@torch.no_grad()
def modify_state_dict_to_dtensor_dict(module: nn.Module, state_dict: dict, prefix: str, strip_keys: bool) -> dict:
    module_state_dict = module.state_dict()

    result = {}
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            stripped_key = key.split(prefix)[1] if strip_keys and prefix != "" else key

            param = module_state_dict[stripped_key]
            device_mesh = param.device_mesh
            placements = param.placements
            result[key] = DTensor.from_local(tensor, device_mesh=device_mesh, placements=placements)

    return result


def get_module_placements(use_padding_free_transformer: bool, sequence_parallel: bool) -> Placement:
    if sequence_parallel:
        if use_padding_free_transformer:
            placement = Shard(0)
        else:
            placement = Shard(1)
    else:
        placement = Replicate()

    return placement
