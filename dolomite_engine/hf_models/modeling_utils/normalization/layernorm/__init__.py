import torch.nn as nn

from .apex import ApexLayerNorm
from .apex_persistent import ApexPersistentLayerNorm


_LAYERNORM_MODULES = {
    "torch": nn.LayerNorm,
    "apex": ApexLayerNorm,
    "apex_persistent": ApexPersistentLayerNorm,
}


def get_layernorm(
    normalized_shape: int,
    eps: float,
    normalization_implementation: str = "torch",
) -> nn.LayerNorm:
    if normalization_implementation in _LAYERNORM_MODULES:
        return _LAYERNORM_MODULES[normalization_implementation](normalized_shape=normalized_shape, eps=eps)

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")
