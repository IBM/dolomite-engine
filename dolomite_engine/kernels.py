from .enums import Kernel


_KERNELS: set[Kernel] = []


def is_kernel_allowed(kernel: Kernel) -> bool:
    return kernel in _KERNELS


def add_kernel(kernel: Kernel) -> None:
    _KERNELS.add(kernel)
