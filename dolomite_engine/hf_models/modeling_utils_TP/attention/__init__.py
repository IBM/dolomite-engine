from ...config import MegatronConfig
from ...enums import AttentionHeadType, PositionEmbeddingType
from .base import Attention_TP
from .flash import FlashAttention2_TP
from .math import MathAttention_TP
from .sdpa import SDPA_TP


_ATTENTION_MODULES = {
    "eager": MathAttention_TP,
    "sdpa": SDPA_TP,
    "flash_attention_2": FlashAttention2_TP,
}


def get_attention_module(
    config: MegatronConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> Attention_TP:
    if use_padding_free_transformer:
        raise NotImplementedError("padding free transformer is not implemented with tensor parallel")

    if attention_implementation in _ATTENTION_MODULES:
        return _ATTENTION_MODULES[attention_implementation](
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            attention_head_type=AttentionHeadType(config.attention_head_type),
            position_embedding_type=PositionEmbeddingType(config.position_embedding_type),
            causal=causal,
            add_bias=config.add_bias,
            scale_attention_weights=config.scale_attn_weights,
            attention_multiplier=config.attention_multiplier,
            attention_softmax_in_fp32=config.attention_softmax_in_fp32,
            scale_attention_softmax_in_fp32=config.scale_attention_softmax_in_fp32,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            layer_idx=layer_idx,
        )

    raise ValueError(f"unexpected `attention_implementation` {attention_implementation}")
