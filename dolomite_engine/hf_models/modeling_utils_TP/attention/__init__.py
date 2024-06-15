from ...config import CommonConfig
from .base import Attention_TP
from .flash import FlashAttention2_TP
from .sdpa import SDPA_TP


_ATTENTION_MODULES = {
    "eager": Attention_TP,
    "sdpa": SDPA_TP,
    "flash_attention_2": FlashAttention2_TP,
}


def get_attention_module(
    config: CommonConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> Attention_TP:
    if use_padding_free_transformer:
        raise NotImplementedError("padding free transformer is not implemented with tensor parallel")

    if attention_implementation in _ATTENTION_MODULES:
        return _ATTENTION_MODULES[attention_implementation](config, causal=causal, layer_idx=layer_idx)

    raise ValueError(f"unexpected `attention_implementation` {attention_implementation}")
