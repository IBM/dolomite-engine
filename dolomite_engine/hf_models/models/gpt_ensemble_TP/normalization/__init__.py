import torch.nn as nn

from .rmsnorm import get_rmsnorm


_NORMALIZATION_FUNCTIONS = {"rmsnorm": get_rmsnorm}


def get_ensemble_normalization_function_TP(
    name: str,
    normalized_shape: int,
    eps: float = 1e-5,
    normalization_implementation: str = "torch",
) -> nn.LayerNorm:
    if name in _NORMALIZATION_FUNCTIONS:
        return _NORMALIZATION_FUNCTIONS[name](
            normalized_shape,
            eps=eps,
            normalization_implementation=normalization_implementation,
        )

    raise ValueError(f"unexpected `normalization_function` {name}")
