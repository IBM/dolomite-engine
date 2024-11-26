from .base import LayerNorm_TP


def get_layernorm(
    normalized_shape: int,
    eps: float,
    use_padding_free_transformer: bool = False,
    sequence_parallel: bool = False,
) -> LayerNorm_TP:
    return LayerNorm_TP(
        normalized_shape=normalized_shape,
        eps=eps,
        use_padding_free_transformer=use_padding_free_transformer,
        sequence_parallel=sequence_parallel,
    )
