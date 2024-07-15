from ...config import CommonConfig
from .base import Attention_TP
from .flash import FlashAttention2_TP
from .padding_free import PaddingFreeAttention_TP
from .sdpa import SDPA_TP


_ATTENTION_MODULES = {
    "eager": Attention_TP,
    "sdpa": SDPA_TP,
    "flash_attention_2": FlashAttention2_TP,
}


def get_attention_module_TP(
    config: CommonConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
    sequence_parallel: bool,
) -> Attention_TP:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = PaddingFreeAttention_TP
    else:
        attention_class = _ATTENTION_MODULES[attention_implementation]

    return attention_class(config, causal=causal, layer_idx=layer_idx, sequence_parallel=sequence_parallel)
