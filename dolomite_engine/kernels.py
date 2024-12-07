import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from .enums import Kernel


_KERNELS: set[Kernel] = set()


def is_kernel_allowed(kernel: Kernel) -> bool:
    return kernel in _KERNELS


def add_kernel(kernel: Kernel) -> None:
    _KERNELS.add(kernel)


class _ACT_BackwardWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, x_grad: AsyncCollectiveTensor) -> torch.Tensor:
        if isinstance(x_grad, AsyncCollectiveTensor):
            x_grad = x_grad.wait()

        return x_grad


def wait_for_ACT(x: torch.Tensor, wait_in_forward: bool, wait_in_backward: bool) -> torch.Tensor:
    if wait_in_forward:
        if isinstance(x, AsyncCollectiveTensor):
            x = x.wait()
    elif wait_in_backward:
        x = _ACT_BackwardWait.apply(x)
    else:
        raise ValueError("either wait_in_forward or wait_in_backward should be True")

    return x
