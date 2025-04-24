from contextlib import contextmanager

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor


_KERNELS = 0


def is_kernel_allowed(kernel) -> bool:
    return kernel in _KERNELS


@contextmanager
def enable_kernels(kernels):
    global _KERNELS

    original_kernels = _KERNELS
    _KERNELS = kernels

    yield

    _KERNELS = original_kernels


@contextmanager
def enable_all_kernels():
    with enable_kernels(1):
        yield


with enable_kernels(3):
    with enable_all_kernels():
        print(_KERNELS)
    print(_KERNELS)
