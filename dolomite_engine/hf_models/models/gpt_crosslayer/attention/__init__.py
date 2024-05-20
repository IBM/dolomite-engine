from ..config import GPTCrossLayerConfig
from .base import CrossLayerAttention, KeyValueProjection
from .flash import CrossLayerFlashAttention2
from .padding_free import CrossLayerPaddingFreeAttention, KeyValuePaddingFreeProjection
from .sdpa import CrossLayerSDPA


_ATTENTION_MODULES = {
    "eager": CrossLayerAttention,
    "sdpa": CrossLayerSDPA,
    "flash_attention_2": CrossLayerFlashAttention2,
}


def get_attention_module(
    config: GPTCrossLayerConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> CrossLayerAttention:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = CrossLayerPaddingFreeAttention
    else:
        attention_class = _ATTENTION_MODULES[attention_implementation]

    return attention_class(config, causal=causal, layer_idx=layer_idx)


_KEY_VALUE_PROJECTION_MODULES = {
    "eager": KeyValueProjection,
    "sdpa": KeyValueProjection,
    "flash_attention_2": KeyValueProjection,
}


def get_key_value_projection(
    config: GPTCrossLayerConfig, attention_implementation: str, use_padding_free_transformer: bool
) -> CrossLayerAttention:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        kv_projection_class = KeyValuePaddingFreeProjection
    else:
        kv_projection_class = _KEY_VALUE_PROJECTION_MODULES[attention_implementation]

    return kv_projection_class(config)
