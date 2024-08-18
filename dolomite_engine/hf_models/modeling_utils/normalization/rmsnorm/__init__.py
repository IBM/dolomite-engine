import torch.nn as nn

from .apex import ApexRMSNorm
from .torchtitan import TorchTitanRMSNorm


_RMSNORM_MODULES = {"torch": nn.RMSNorm, "apex": ApexRMSNorm, "torchtitan": TorchTitanRMSNorm}


def get_rmsnorm(normalized_shape: int, eps: float, normalization_implementation: str = "torch") -> nn.RMSNorm:
    if normalization_implementation in _RMSNORM_MODULES:
        return _RMSNORM_MODULES[normalization_implementation](normalized_shape=normalized_shape, eps=eps)

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")
