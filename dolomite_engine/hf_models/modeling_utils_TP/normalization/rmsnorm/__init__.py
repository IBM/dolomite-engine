from .base import RMSNorm_TP


_RMSNORM_MODULES = {"torch": RMSNorm_TP}


def get_rmsnorm(
    normalized_shape: int,
    eps: float,
    use_padding_free_transformer: bool = False,
    sequence_parallel: bool = False,
) -> RMSNorm_TP:
    return RMSNorm_TP(
        normalized_shape=normalized_shape,
        eps=eps,
        use_padding_free_transformer=use_padding_free_transformer,
        sequence_parallel=sequence_parallel,
    )
