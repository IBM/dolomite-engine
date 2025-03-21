from ....enums import AttentionHeadType, PositionEmbeddingType
from ..config import GPTCrossLayerConfig
from .base import CrossLayerAttention, KeyValueProjection
from .padding_free import CrossLayerPaddingFreeAttention


def get_sequence_mixer(
    config: GPTCrossLayerConfig, causal: bool, use_padding_free_transformer: bool, layer_idx: int
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
        attention_class = CrossLayerPaddingFreeAttention
    else:
        attention_class = CrossLayerAttention

    return attention_class(**sequence_mixer_kwargs)
