from ...ladder_residual import LadderResidualConfig
from .padding_free import LadderResidualPaddingFreeAttention_TP


def get_attention_module_TP(
    config: LadderResidualConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
    sequence_parallel: bool,
) -> LadderResidualPaddingFreeAttention_TP:
    assert use_padding_free_transformer

    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = LadderResidualPaddingFreeAttention_TP

    return attention_class(config, causal=causal, layer_idx=layer_idx, sequence_parallel=sequence_parallel)
