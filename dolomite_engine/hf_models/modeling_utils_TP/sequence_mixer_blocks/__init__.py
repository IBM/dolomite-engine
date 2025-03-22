from ...config import CommonConfig
from ...enums import PositionEmbeddingType
from .softmax_attention import Attention_TP


def get_sequence_mixer_TP(
    config: CommonConfig,
    causal: bool,
    use_padding_free_transformer: bool,
    layer_idx: int,
    sequence_parallel: bool,
) -> Attention_TP:
    block = config.sequence_mixer_blocks[layer_idx]
    sequence_mixer_type = block.sequence_mixer_type

    sequence_mixer_kwargs = dict(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=block.num_key_value_heads,
        attention_multiplier=block.attention_multiplier,
        attention_head_type=block.attention_head_type,
        position_embedding_type=PositionEmbeddingType(config.position_embedding_type),
        add_bias=block.add_bias,
        softmax_dropout=block.softmax_dropout,
        dropout=block.dropout,
        init_method=config.init_method,
        initializer_range=config.initializer_range,
        m_width=config.m_width,
        num_layers=config.num_layers,
        causal=causal,
        layer_idx=layer_idx,
        sequence_parallel=sequence_parallel,
    )

    if sequence_mixer_type == "softmax_attention":
        return Attention_TP(**sequence_mixer_kwargs, use_padding_free_transformer=use_padding_free_transformer)
