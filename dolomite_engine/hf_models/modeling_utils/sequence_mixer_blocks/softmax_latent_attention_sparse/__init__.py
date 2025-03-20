from ....config import CommonConfig
from .base import Attention
from .flash import FlashAttention2
from .padding_free import PaddingFreeAttention
from .sdpa import SDPA


_ATTENTION_MODULES = {
    "eager": Attention,
    "sdpa": SDPA,
    "flash_attention_2": FlashAttention2,
}


def get_attention_module(
    config: CommonConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> Attention:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = PaddingFreeAttention
    else:
        attention_class = _ATTENTION_MODULES[attention_implementation]

    return attention_class(config, causal=causal, layer_idx=layer_idx)
