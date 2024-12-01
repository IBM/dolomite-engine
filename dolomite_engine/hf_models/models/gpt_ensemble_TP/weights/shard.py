from .....utils import SafeTensorsWeightsManager
from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils_TP import tensor_parallel_split_safetensor_slice
from ...gpt_dolomite_TP.weights.shard import _get_embeddings_or_lm_head, _get_layernorm
from ...gpt_ensemble import GPTEnsembleConfig


def get_gpt_ensemble_model_parallel_state_dict(
    config: GPTEnsembleConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    tensor_parallel_word_embeddings: bool,
    num_pipeline_stages: int,
    pipeline_stage_id: int,
) -> dict:
    assert num_pipeline_stages == 1

    # word embeddings
    state_dict = _get_embeddings_or_lm_head(
        safetensors_weights_manager,
        prefix="transformer.wte.",
        vocab_size=config.vocab_size,
        tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
    )

    # positional embeddings
    if PositionEmbeddingType(config.position_embedding_type) == PositionEmbeddingType.learned_absolute:
        state_dict.update(
            _get_embeddings_or_lm_head(
                safetensors_weights_manager,
                prefix="transformer.wpe.",
                vocab_size=config.n_positions,
                tensor_parallel_word_embeddings=False,
            )
        )

    for layer_idx in range(config.n_layer):
        prefix = f"transformer.h.{layer_idx}."

        if layer_idx == 0 or config.reduce_pattern[layer_idx - 1]["mlp"]:
            state_dict.update(_get_layernorm(safetensors_weights_manager, prefix=prefix + "ln_1."))
        else:
            state_dict.update(_get_ensemble_layernorm(safetensors_weights_manager, prefix=prefix + "ln_1."))

        state_dict.update(
            _get_attention(
                config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "attn."
            )
        )

        if config.reduce_pattern[layer_idx]["attention"]:
            state_dict.update(_get_layernorm(safetensors_weights_manager, prefix=prefix + "ln_2."))
        else:
            state_dict.update(_get_ensemble_layernorm(safetensors_weights_manager, prefix=prefix + "ln_2."))

        state_dict.update(
            _get_mlp(config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "mlp.")
        )

    state_dict.update(_get_layernorm(safetensors_weights_manager, prefix="transformer.ln_f."))

    if not config.tie_word_embeddings:
        state_dict.update(
            _get_embeddings_or_lm_head(
                safetensors_weights_manager=safetensors_weights_manager,
                prefix="lm_head.",
                vocab_size=config.vocab_size,
                tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
            )
        )

    return state_dict


def _get_ensemble_layernorm(safetensors_weights_manager: SafeTensorsWeightsManager, prefix: str) -> dict:
    weight = safetensors_weights_manager.get_slice(prefix + "weight")
    state_dict = {prefix + "weight": tensor_parallel_split_safetensor_slice(weight, dim=0)}

    assert not safetensors_weights_manager.has_tensor(prefix + "bias")

    return state_dict


def _get_attention(
    config: GPTEnsembleConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
) -> None:
    state_dict = {}

    if AttentionHeadType(config.attention_head_type) == AttentionHeadType.mqa:
        raise ValueError("GPTEnsemble doesn't support mqa")
    else:
        state_dict.update(
            _get_column_parallel(
                config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "c_attn."
            )
        )

    state_dict.update(
        _get_row_parallel(
            config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "c_proj."
        )
    )

    return state_dict


def _get_mlp(
    config: GPTEnsembleConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
) -> None:
    state_dict = _get_column_parallel(
        config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "c_fc."
    )

    state_dict.update(
        _get_row_parallel(
            config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "c_proj."
        )
    )

    return state_dict


def _get_column_parallel(
    config: GPTEnsembleConfig, safetensors_weights_manager: SafeTensorsWeightsManager, prefix: str
) -> dict:
    weight = safetensors_weights_manager.get_slice(prefix + "weight")
    weight = tensor_parallel_split_safetensor_slice(weight, dim=0)
    state_dict = {prefix + "weight": weight.T}

    if config.add_bias:
        bias = safetensors_weights_manager.get_slice(prefix + "bias")
        bias = tensor_parallel_split_safetensor_slice(bias, dim=0)
        state_dict[prefix + "bias"] = bias

    return state_dict


def _get_row_parallel(
    config: GPTEnsembleConfig, safetensors_weights_manager: SafeTensorsWeightsManager, prefix: str
) -> dict:
    weight = safetensors_weights_manager.get_slice(prefix + "weight")
    weight = tensor_parallel_split_safetensor_slice(weight, dim=0)
    state_dict = {prefix + "weight": weight.T}

    if config.add_bias:
        state_dict[prefix + "bias"] = safetensors_weights_manager.get_tensor(prefix + "bias")

    return state_dict
