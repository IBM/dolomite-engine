from ...ladder_residual import LadderResidualConfig
from .flash import LadderResidualFlashAttention2_TP
from .padding_free import PaddingFreeAttention_TP
from .sdpa import LadderResidualSDPA_TP


_ATTENTION_MODULES = {"sdpa": LadderResidualSDPA_TP, "flash_attention_2": LadderResidualFlashAttention2_TP}


def get_attention_module_TP(
    config: LadderResidualConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
    sequence_parallel: bool,
) -> LadderResidualSDPA_TP | LadderResidualFlashAttention2_TP:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = PaddingFreeAttention_TP
    else:
        attention_class = _ATTENTION_MODULES[attention_implementation]

    return attention_class(config, causal=causal, layer_idx=layer_idx, sequence_parallel=sequence_parallel)
