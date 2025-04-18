from ...config import CommonConfig
from .mamba2 import Mamba2
from .multihead_latent_attention import MultiHeadLatentAttention
from .rnn import RNN
from .softmax_attention import (
    Attention,
    get_attention_head_type,
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


def get_sequence_mixer(
    config: CommonConfig, causal: bool, use_padding_free_transformer: bool, layer_idx: int, is_mtp_block: bool = False
) -> Attention | Mamba2:
    if is_mtp_block:
        block = config.mtp_blocks[layer_idx].sequence_mixer
    else:
        block = config.sequence_mixer_blocks[layer_idx]

    sequence_mixer_type = block.sequence_mixer_type

    if sequence_mixer_type == "mamba2":
        return Mamba2(
            hidden_size=config.hidden_size,
            ssm_state_size=block.state_size,
            ssm_intermediate_size=block.intermediate_size,
            ssm_num_heads=block.num_heads,
            conv_kernel_size=block.conv_kernel_size,
            time_step_limit=block.time_step_limit,
            add_bias=block.add_bias,
            use_conv_bias=block.use_conv_bias,
            ssm_activation_function=block.activation_function,
            num_groups=block.num_groups,
            chunk_size=block.chunk_size,
            layer_norm_epsilon=config.layer_norm_epsilon,
            initializer_range=config.initializer_range,
            init_method=config.init_method,
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
        )
    elif sequence_mixer_type == "multihead_latent_attention":
        return MultiHeadLatentAttention(
            hidden_size=config.hidden_size,
            query_compression_size=block.query_compression_size,
            key_value_compression_size=block.key_value_compression_size,
            num_attention_heads=block.num_attention_heads,
            attention_multiplier=block.attention_multiplier,
            position_embedding_type=config.position_embedding_type,
            add_bias=block.add_bias,
            softmax_dropout=block.softmax_dropout,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            causal=True,
            layer_idx=layer_idx,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif sequence_mixer_type == "rnn":
        return RNN(
            input_size=config.hidden_size,
            state_size=block.state_size,
            output_size=config.hidden_size,
            num_heads=block.num_heads,
            m_width=config.m_width,
            num_layers=config.num_layers,
            add_bias=block.add_bias,
            initializer_range=config.initializer_range,
            init_method=config.init_method,
            gradient_clipping=block.gradient_clipping,
        )
    else:
        sequence_mixer_kwargs = dict(
            hidden_size=config.hidden_size,
            num_attention_heads=block.num_attention_heads,
            num_key_value_heads=block.num_key_value_heads,
            attention_multiplier=block.attention_multiplier,
            position_embedding_type=config.position_embedding_type,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            causal=causal,
            layer_idx=layer_idx,
        )

        if sequence_mixer_type == "softmax_attention":
            return Attention(
                **sequence_mixer_kwargs,
                softmax_dropout=block.softmax_dropout,
                use_padding_free_transformer=use_padding_free_transformer,
            )
        elif sequence_mixer_type == "stickbreaking_attention":
            if use_padding_free_transformer:
                return PaddingFreeSBAttention(**sequence_mixer_kwargs)
            else:
                return SBAttention(**sequence_mixer_kwargs)
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({sequence_mixer_type})")
