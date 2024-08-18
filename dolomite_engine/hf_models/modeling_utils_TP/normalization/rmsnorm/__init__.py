from .base import RMSNorm_TP


_RMSNORM_MODULES = {"torch": RMSNorm_TP}


def get_rmsnorm(
    normalized_shape: int,
    eps: float,
    normalization_implementation: str = "torch",
    use_padding_free_transformer: bool = False,
    sequence_parallel: bool = False,
) -> RMSNorm_TP:
    if normalization_implementation in _RMSNORM_MODULES:
        return _RMSNORM_MODULES[normalization_implementation](
            normalized_shape=normalized_shape,
            eps=eps,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")
