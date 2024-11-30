import torch
import torch.distributed
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard

from ...utils import ProcessGroupManager, divide_if_divisible


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
        output = slice[:, :, start_index:end_index]

    return output


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
