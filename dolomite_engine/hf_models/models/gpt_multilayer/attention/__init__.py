from ..config import GPTMultiLayerConfig
from .base import KeyValueProjection, MultiLayerAttention
from .flash import MultiLayerFlashAttention2
from .math import MultiLayerMathAttention
from .padding_free import KeyValuePaddingFreeProjection, MultiLayerPaddingFreeAttention
from .sdpa import MultiLayerSDPA


_ATTENTION_MODULES = {
    "eager": MultiLayerMathAttention,
    "sdpa": MultiLayerSDPA,
    "flash_attention_2": MultiLayerFlashAttention2,
}


def get_attention_module(
    config: GPTMultiLayerConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> MultiLayerAttention:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = MultiLayerPaddingFreeAttention
    else:
        attention_class = _ATTENTION_MODULES[attention_implementation]

    return attention_class(config, causal=causal, layer_idx=layer_idx)


_KEY_VALUE_PROJECTION_MODULES = {
    "eager": KeyValueProjection,
    "sdpa": KeyValueProjection,
    "flash_attention_2": KeyValueProjection,
}


def get_key_value_projection(
    config: GPTMultiLayerConfig, attention_implementation: str, use_padding_free_transformer: bool
) -> MultiLayerAttention:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        kv_projection_class = KeyValuePaddingFreeProjection
    else:
        kv_projection_class = _KEY_VALUE_PROJECTION_MODULES[attention_implementation]

    return kv_projection_class(config)
