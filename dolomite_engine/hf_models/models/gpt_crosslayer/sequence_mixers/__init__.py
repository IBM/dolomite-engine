from ....enums import AttentionHeadType, PositionEmbeddingType
from ..config import GPTCrossLayerConfig
from .base import CrossLayerAttention, KeyValueProjection
from .flash import CrossLayerFlashAttention2
from .padding_free import CrossLayerPaddingFreeAttention, KeyValuePaddingFreeProjection


_ATTENTION_MODULES = {
    "eager": CrossLayerAttention,
    "sdpa": CrossLayerAttention,
    "flash_attention_2": CrossLayerFlashAttention2,
}


def get_sequence_mixer(
    config: GPTCrossLayerConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> CrossLayerAttention:
    block = config.sequence_mixer_blocks[layer_idx]
    assert block.sequence_mixer_type == "softmax_attention"

    sequence_mixer_kwargs = dict(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=block.num_key_value_heads,
        attention_multiplier=block.attention_multiplier,
        attention_head_type=AttentionHeadType(block.attention_head_type),
        position_embedding_type=PositionEmbeddingType(config.position_embedding_type),
        add_bias=block.add_bias,
        softmax_dropout=block.softmax_dropout,
        dropout=block.dropout,
        initializer_range=config.initializer_range,
        num_layers=config.num_layers,
        causal=causal,
        layer_idx=layer_idx,
    )

    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = CrossLayerPaddingFreeAttention
    else:
        attention_class = _ATTENTION_MODULES[attention_implementation]

    return attention_class(**sequence_mixer_kwargs)


_KEY_VALUE_PROJECTION_MODULES = {
    "eager": KeyValueProjection,
    "sdpa": KeyValueProjection,
    "flash_attention_2": KeyValueProjection,
}


def get_key_value_projection(
    config: GPTCrossLayerConfig, attention_implementation: str, use_padding_free_transformer: bool, layer_idx: int
) -> CrossLayerAttention:
    kwargs = dict(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.sequence_mixer_blocks[layer_idx].num_key_value_heads,
        add_bias=config.sequence_mixer_blocks[layer_idx].add_bias,
        initializer_range=config.initializer_range,
        normalization_function=config.normalization_function,
        layer_norm_epsilon=config.layer_norm_epsilon,
    )

    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        kv_projection_class = KeyValuePaddingFreeProjection
    else:
        kv_projection_class = _KEY_VALUE_PROJECTION_MODULES[attention_implementation]

    return kv_projection_class(**kwargs)
