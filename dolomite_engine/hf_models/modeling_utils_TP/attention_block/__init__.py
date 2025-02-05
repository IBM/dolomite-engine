from ...config import CommonConfig
from .softmax_attention import SDPA_TP, Attention_TP, FlashAttention2_TP, PaddingFreeAttention_TP


_ATTENTION_MODULES = {
    "eager": Attention_TP,
    "sdpa": SDPA_TP,
    "flash_attention_2": FlashAttention2_TP,
}


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
    attention_block_type = config.attention_blocks[layer_idx]["attention_block_type"]

    if attention_block_type == "softmax_attention":
        if use_padding_free_transformer:
            assert (
                attention_implementation == "flash_attention_2"
            ), "padding free transformer only works with flash attention"
            return PaddingFreeAttention_TP(config, causal=causal, layer_idx=layer_idx)
        else:
            return _ATTENTION_MODULES[attention_implementation](config, causal=causal, layer_idx=layer_idx)
