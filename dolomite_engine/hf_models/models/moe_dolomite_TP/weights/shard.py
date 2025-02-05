from .....utils import SafeTensorsWeightsManager, divide_if_divisible
from ....enums import PositionEmbeddingType
from ...gpt_dolomite_TP.weights.shard import _get_attention, _get_embeddings_or_lm_head, _get_layernorm, _get_mlp


def get_moe_dolomite_tensor_parallel_state_dict(
    config,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    num_pipeline_stages: int,
    pipeline_stage_id: int,
) -> dict:
    is_first_pipeline_stage = pipeline_stage_id == 0
    is_last_pipeline_stage = pipeline_stage_id == num_pipeline_stages - 1

    layers_per_stage = divide_if_divisible(
        config.n_layer, num_pipeline_stages, "layers should be divisible by num_pipeline_stages"
    )

    layer_start_id = layers_per_stage * pipeline_stage_id
    layer_end_id = layers_per_stage * (pipeline_stage_id + 1)

    state_dict = {}

    if is_first_pipeline_stage:
        # word embeddings
        state_dict.update(
            _get_embeddings_or_lm_head(
                safetensors_weights_manager, prefix="transformer.wte.", vocab_size=config.vocab_size
            )
        )

        # positional embeddings
        if PositionEmbeddingType(config.position_embedding_type) == PositionEmbeddingType.learned_absolute:
            state_dict.update(
                _get_embeddings_or_lm_head(
                    safetensors_weights_manager, prefix="transformer.wpe.", vocab_size=config.n_positions
                )
            )

    for layer_idx in range(layer_start_id, layer_end_id):
        prefix = f"transformer.h.{layer_idx}."

        state_dict.update(_get_layernorm(safetensors_weights_manager, prefix=prefix + "ln_1."))

        state_dict.update(
            _get_attention(
                config=config,
                safetensors_weights_manager=safetensors_weights_manager,
                prefix=prefix + "attn.",
                column_parallel_shard_dim=0,
                row_parallel_shard_dim=1,
            )
        )

        state_dict.update(_get_layernorm(safetensors_weights_manager, prefix=prefix + "ln_2."))

        state_dict.update(
            _get_moe(
                config=config,
                safetensors_weights_manager=safetensors_weights_manager,
                prefix=prefix + "mlp.",
                column_parallel_shard_dim=1,
                row_parallel_shard_dim=2,
            )
        )

    if is_last_pipeline_stage:
        state_dict.update(_get_layernorm(safetensors_weights_manager, prefix="transformer.ln_f."))

        if not config.tie_word_embeddings:
            state_dict.update(
                _get_embeddings_or_lm_head(
                    safetensors_weights_manager=safetensors_weights_manager,
                    prefix="lm_head.",
                    vocab_size=config.vocab_size,
                )
            )

    return state_dict


def _get_moe(
    config,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
    column_parallel_shard_dim: int,
    row_parallel_shard_dim: int,
) -> None:
    state_dict = {prefix + "gate.weight": safetensors_weights_manager.get_tensor(prefix + "gate.weight")}

    assert not config.add_bias

    state_dict.update(
        _get_mlp(
            config=config,
            safetensors_weights_manager=safetensors_weights_manager,
            prefix=prefix,
            column_parallel_shard_dim=column_parallel_shard_dim,
            row_parallel_shard_dim=row_parallel_shard_dim,
        )
    )

    return state_dict
