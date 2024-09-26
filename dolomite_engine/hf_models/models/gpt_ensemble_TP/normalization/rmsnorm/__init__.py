import torch.nn as nn

from .base import EnsembleRMSNorm_TP


_RMSNORM_MODULES = {"torch": EnsembleRMSNorm_TP}


def get_rmsnorm(
    normalized_shape: int, tp_world_size: int, eps: float, normalization_implementation: str = "torch"
) -> nn.LayerNorm:
    if normalization_implementation in _RMSNORM_MODULES:
        return _RMSNORM_MODULES[normalization_implementation](
            normalized_shape=normalized_shape, tp_world_size=tp_world_size, eps=eps
        )

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")
