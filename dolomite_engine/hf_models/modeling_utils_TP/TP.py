import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard

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
    tensor: torch.Tensor,
    current_placement: Placement,
    desired_placement: Placement | None = None,
    dtensor_input_allowed: bool = False,
) -> DTensor:
    # if already a tensor, we return as-is
    if isinstance(tensor, DTensor):
        if dtensor_input_allowed:
            return tensor
        else:
            raise ValueError("input is already a DTensor")

    tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

    dtensor = DTensor.from_local(tensor, device_mesh=tp_mesh, run_check=False, placements=[current_placement])
    if desired_placement is not None:
        dtensor = dtensor.redistribute(device_mesh=tp_mesh, placements=[desired_placement], async_op=True)

    return dtensor


def dtensor_to_tensor(
    dtensor: DTensor,
    desired_placement: Placement | None = None,
    grad_placement: Placement | None = None,
    tensor_input_allowed: bool = False,
) -> torch.Tensor:
    if not isinstance(dtensor, DTensor):
        assert isinstance(dtensor, torch.Tensor), "invalid type found"

        if tensor_input_allowed:
            return dtensor
        else:
            raise ValueError("input is already a Tensor")

    if desired_placement is not None:
        dtensor = dtensor.redistribute(
            device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[desired_placement], async_op=True
        )

    tensor = dtensor.to_local(grad_placements=None if grad_placement is None else [grad_placement])

    return tensor


@torch.no_grad()
def modify_state_dict_to_dtensor_dict(module: nn.Module, state_dict: dict, prefix: str, strip_keys: bool) -> dict:
    result = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, DTensor):
            continue

        if key.startswith(prefix):
            striped_key = key.split(prefix)[1] if strip_keys else key

            device_mesh = getattr(module, striped_key).device_mesh
            placements = getattr(module, striped_key).placements
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


def _tensor_parallel_all_reduce(x: torch.Tensor) -> torch.Tensor:
    if ProcessGroupManager.get_tensor_parallel_world_size() == 1:
        return x

    torch.distributed.all_reduce(x, group=ProcessGroupManager.get_tensor_parallel_group())

    return x


def _tensor_parallel_all_gather(x: torch.Tensor, dim: int) -> torch.Tensor:
    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

    if tp_world_size == 1:
        return x

    original_shape = list(x.size())
    original_shape[0] = original_shape[0] * tp_world_size

    output = torch.empty(original_shape, device=x.device, dtype=x.dtype)
    torch.distributed.all_gather_into_tensor(output, x, group=ProcessGroupManager.get_tensor_parallel_group())

    if dim != 0:
        output = output.chunk(tp_world_size)
        output = torch.cat(output, dim=dim)

    return output


def _tensor_parallel_reduce_scatter(x: torch.Tensor, dim: int) -> torch.Tensor:
    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

    if tp_world_size == 1:
        return x

    if dim != 0:
        x = x.chunk(tp_world_size, dim=dim)
        x = torch.cat(x)

    original_shape = list(x.size())
    original_shape[0] = original_shape[0] // tp_world_size

    output = torch.empty(original_shape, device=x.device, dtype=x.dtype)
    torch.distributed.reduce_scatter_tensor(output, x, group=ProcessGroupManager.get_tensor_parallel_group())

    return output


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> torch.Tensor:
        return _tensor_parallel_all_reduce(x_grad)


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _tensor_parallel_all_reduce(x)

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> torch.Tensor:
        return x_grad


class _AllGatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        return _tensor_parallel_all_gather(x, dim=dim)

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> torch.Tensor:
        dim = ctx.dim
        return _tensor_parallel_reduce_scatter(x_grad, dim=dim), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        return _tensor_parallel_reduce_scatter(x, dim=dim)

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> torch.Tensor:
        dim = ctx.dim
        return _tensor_parallel_all_gather(x_grad, dim=dim), None


def copy_to_tensor_parallel_region(x: torch.Tensor) -> torch.Tensor:
    return _CopyToTensorParallelRegion.apply(x)


def reduce_from_tensor_parallel_region(x: torch.Tensor) -> torch.Tensor:
    return _ReduceFromTensorParallelRegion.apply(x)


def all_gather_from_sequence_parallel_region(x: torch.Tensor, dim: int) -> torch.Tensor:
    return _AllGatherFromSequenceParallelRegion.apply(x, dim)


def reduce_scatter_to_sequence_parallel_region(x: torch.Tensor, dim: int) -> torch.Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(x, dim)
