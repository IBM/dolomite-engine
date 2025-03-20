from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...config import CommonConfig
from ...enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from .mamba2 import Mamba2Base, Mamba2CUDA
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


def get_sequence_mixer(
    config: CommonConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> Attention | Mamba2Base:
    block = config.sequence_mixer_blocks[layer_idx]
    sequence_mixer_type = block.sequence_mixer_type

    if sequence_mixer_type == "mamba2":
        mamba_class = Mamba2CUDA if is_kernel_allowed(Kernel.mamba2_ssm) else Mamba2Base
        return mamba_class(
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
            init_method=InitMethod(config.init_method),
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
        )
    else:
        sequence_mixer_kwargs = dict(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=block.num_key_value_heads,
            attention_multiplier=block.attention_multiplier,
            attention_head_type=AttentionHeadType(block.attention_head_type),
            position_embedding_type=PositionEmbeddingType(config.position_embedding_type),
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=InitMethod(config.init_method),
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            causal=causal,
            layer_idx=layer_idx,
        )

        # Add latent and sparse attention parameters
        if hasattr(block, "use_latent_attention"):
            sequence_mixer_kwargs["use_latent_attention"] = block.use_latent_attention
            if block.use_latent_attention:
                sequence_mixer_kwargs["kv_compression_dim"] = block.kv_compression_dim
                sequence_mixer_kwargs["head_dim_latent"] = block.head_dim_latent

        if hasattr(block, "use_sparse_attention"):
            sequence_mixer_kwargs["use_sparse_attention"] = block.use_sparse_attention
            if block.use_sparse_attention:
                sequence_mixer_kwargs["sparse_pattern"] = block.sparse_pattern
                sequence_mixer_kwargs["moba_chunk_size"] = getattr(block, "moba_chunk_size", 1024)
                sequence_mixer_kwargs["moba_topk"] = getattr(block, "moba_topk", 8)

        if sequence_mixer_type == "softmax_attention":
            sequence_mixer_kwargs["softmax_dropout"] = block.softmax_dropout

            if use_padding_free_transformer:
                assert (
                    attention_implementation == "flash_attention_2"
                ), "padding free transformer only works with flash attention"
                return PaddingFreeAttention(**sequence_mixer_kwargs)
            elif block.use_sparse_attention:
                # Use sparse attention implementation
                from .softmax_latent_attention_sparse import _ATTENTION_MODULES as SPARSE_MODULES

                return SPARSE_MODULES[attention_implementation](**sequence_mixer_kwargs)
            elif block.use_latent_attention:
                # Use latent attention implementation
                from .softmax_latent_attention import _ATTENTION_MODULES as LATENT_MODULES

                # Create a copy and remove sparse-specific parameters
                latent_kwargs = sequence_mixer_kwargs.copy()
                latent_kwargs.pop("use_sparse_attention", None)
                latent_kwargs.pop("sparse_pattern", None)
                latent_kwargs.pop("moba_chunk_size", None)
                latent_kwargs.pop("moba_topk", None)
                return LATENT_MODULES[attention_implementation](**latent_kwargs)
            else:
                # Filter out latent and sparse parameters for regular attention
                regular_kwargs = {
                    k: v
                    for k, v in sequence_mixer_kwargs.items()
                    if k
                    not in [
                        "use_latent_attention",
                        "kv_compression_dim",
                        "use_sparse_attention",
                        "head_dim_latent",
                        "sparse_pattern",
                        "moba_chunk_size",
                        "moba_topk",
                    ]
                }
                return _ATTENTION_MODULES[attention_implementation](**regular_kwargs)
        elif sequence_mixer_type == "stickbreaking_attention":
            if use_padding_free_transformer:
                return PaddingFreeSBAttention(**sequence_mixer_kwargs)
            else:
                return SBAttention(**sequence_mixer_kwargs)
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({sequence_mixer_type})")
