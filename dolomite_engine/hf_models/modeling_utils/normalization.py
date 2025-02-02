import torch
import torch.nn as nn

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...utils import is_cute_kernels_available


if is_cute_kernels_available():
    from cute_kernels import rmsnorm_cute


_NORMALIZATION_FUNCTIONS = {"layernorm": nn.LayerNorm, "rmsnorm": nn.RMSNorm}


class CuteRMSNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_cute(x=x, weight=self.weight, eps=self.eps, memory_efficient=False)


def get_normalization_function(
    normalization_function: str, normalized_shape: int, eps: float = 1e-5
) -> nn.LayerNorm | nn.RMSNorm:
    if is_kernel_allowed(Kernel.cute_rmsnorm) and normalization_function == "rmsnorm":
        normalization = CuteRMSNorm(normalized_shape, eps=eps)
    else:
        if normalization_function in _NORMALIZATION_FUNCTIONS:
            normalization = _NORMALIZATION_FUNCTIONS[normalization_function](normalized_shape, eps=eps)
        else:
            raise ValueError(f"unexpected `normalization_function` {normalization_function}")

    return normalization
