from ....config import CommonConfig
from .base import MLAAttention
from .flash import MLAFlashAttention2
from .padding_free import MLAPaddingFreeAttention
from .sdpa import MLASDPA


_ATTENTION_MODULES = {
    "eager": MLAAttention,
    "sdpa": MLASDPA,
    "flash_attention_2": MLAFlashAttention2,
}


def get_attention_module(
    config: CommonConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> MLAAttention:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = MLAPaddingFreeAttention
    else:
        attention_class = _ATTENTION_MODULES[attention_implementation]

    return attention_class(config, causal=causal, layer_idx=layer_idx)
