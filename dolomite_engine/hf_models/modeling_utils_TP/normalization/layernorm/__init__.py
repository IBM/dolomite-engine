from .base import LayerNorm_TP


_LAYERNORM_MODULES = {"torch": LayerNorm_TP}


def get_layernorm(
    normalized_shape: int,
    eps: float,
    normalization_implementation: str = "torch",
) -> LayerNorm_TP:
    if normalization_implementation in _LAYERNORM_MODULES:
        return _LAYERNORM_MODULES[normalization_implementation](normalized_shape=normalized_shape, eps=eps)

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")
