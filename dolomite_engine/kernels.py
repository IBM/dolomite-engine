from .enums import Kernel


_KERNELS: set[Kernel] = set()


def is_kernel_allowed(kernel: Kernel) -> bool:
    return kernel in _KERNELS


def add_kernel(kernel: Kernel) -> None:
    _KERNELS.add(kernel)
