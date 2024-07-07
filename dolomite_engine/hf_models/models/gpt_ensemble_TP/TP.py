import torch
import torch.distributed
from torch.distributed import ReduceOp

from ....utils import ProcessGroupManager


def ensemble_reduce_from_tensor_parallel_region(input: torch.Tensor) -> torch.Tensor:
    return _EnsembleReduceFromTensorParallelRegion.apply(input)


class _EnsembleReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return _ensemble_tensor_parallel_all_reduce(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def _ensemble_tensor_parallel_all_reduce(input: torch.Tensor) -> torch.Tensor:
    if ProcessGroupManager.get_tensor_parallel_world_size() == 1:
        return input

    torch.distributed.all_reduce(input, op=ReduceOp.AVG, group=ProcessGroupManager.get_tensor_parallel_group())

    return input
