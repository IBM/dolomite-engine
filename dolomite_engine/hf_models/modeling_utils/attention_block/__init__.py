from ...config import CommonConfig
from .softmax_attention import (
    SDPA,
    Attention,
    FlashAttention2,
    PaddingFreeAttention,
    get_attention_module,
    interleave_query_key_value_tensor_for_attention,
    interleave_query_key_value_tensor_for_gqa,
    interleave_query_key_value_tensor_for_mha,
    interleave_query_key_value_tensor_for_mqa,
    repeat_key_value,
    split_query_key_value_tensor_for_attention,
    split_query_key_value_tensor_for_gqa,
    split_query_key_value_tensor_for_mha,
    split_query_key_value_tensor_for_mqa,
)
from .stickbreaking_attention import PaddingFreeSBAttention, SBAttention


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
    attention_block_type = config.attention_blocks[layer_idx]["attention_block_type"]

    if attention_block_type == "softmax_attention":
        if use_padding_free_transformer:
            assert (
                attention_implementation == "flash_attention_2"
            ), "padding free transformer only works with flash attention"
            return PaddingFreeAttention(config, causal=causal, layer_idx=layer_idx)
        else:
            return _ATTENTION_MODULES[attention_implementation]
    elif attention_block_type == "stickbreaking_attention":
        if use_padding_free_transformer:
            return PaddingFreeSBAttention(config, causal=True, layer_idx=layer_idx)
        else:
            return SBAttention(config, causal=True, layer_idx=layer_idx)
