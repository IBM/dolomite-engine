import torch

from .....utils import ProcessGroupManager, SafeTensorsWeightsManager
from ....enums import PositionEmbeddingType
from ....modeling_utils import is_glu
from ....modeling_utils_TP import tensor_parallel_split_safetensor_slice
from ....utils import divide_if_divisible
from ...gpt_dolomite_TP.weights.shard import (
    _get_attention_weights,
    _get_column_parallel_weights,
    _get_mlp_weights,
    _get_row_parallel_weights,
    _get_word_embedding_weights,
)
from ...moe_dolomite import MoEDolomiteConfig


def get_moe_dolomite_tensor_parallel_state_dict(
    config: MoEDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    tensor_parallel_word_embeddings: bool,
) -> dict:
    # word embeddings
    state_dict = _get_word_embedding_weights(
        safetensors_weights_manager,
        prefix="transformer.wte.",
        vocab_size=config.vocab_size,
        tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
    )

    # positional embeddings
    if PositionEmbeddingType(config.position_embedding_type) == PositionEmbeddingType.learned_absolute:
        state_dict.update(
            _get_word_embedding_weights(
                safetensors_weights_manager,
                prefix="transformer.wpe.",
                vocab_size=config.n_positions,
                tensor_parallel_word_embeddings=False,
            )
        )

    for layer_idx in range(config.n_layer):
        prefix = f"transformer.h.{layer_idx}."

        state_dict.update({prefix + "ln_1.weight": safetensors_weights_manager.get_tensor(prefix + "ln_1.weight")})
        if safetensors_weights_manager.has_tensor(prefix + "ln_1.bias"):
            state_dict.update({prefix + "ln_1.bias": safetensors_weights_manager.get_tensor(prefix + "ln_1.bias")})

        state_dict.update(
            _get_attention_weights(
                config=config,
                safetensors_weights_manager=safetensors_weights_manager,
                prefix=prefix + "attn.",
                column_parallel_shard_dim=0,
                row_parallel_shard_dim=1,
            )
        )

        state_dict.update({prefix + "ln_2.weight": safetensors_weights_manager.get_tensor(prefix + "ln_2.weight")})
        if safetensors_weights_manager.has_tensor(prefix + "ln_2.bias"):
            state_dict.update({prefix + "ln_2.bias": safetensors_weights_manager.get_tensor(prefix + "ln_2.bias")})

        state_dict.update(
            _get_moe_weights(
                config=config,
                safetensors_weights_manager=safetensors_weights_manager,
                prefix=prefix + "moe.",
                column_parallel_shard_dim=0,
                row_parallel_shard_dim=2,
            )
        )

    state_dict.update({"transformer.ln_f.weight": safetensors_weights_manager.get_tensor("transformer.ln_f.weight")})
    if safetensors_weights_manager.has_tensor("transformer.ln_f.bias"):
        state_dict.update({"transformer.ln_f.bias": safetensors_weights_manager.get_tensor("transformer.ln_f.bias")})

    if not config.tie_word_embeddings:
        state_dict.update(
            _get_word_embedding_weights(
                safetensors_weights_manager=safetensors_weights_manager,
                prefix="lm_head.",
                vocab_size=config.vocab_size,
                tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
            )
        )

    return state_dict


def _get_moe_weights(
    config: MoEDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
    column_parallel_shard_dim: int,
    row_parallel_shard_dim: int,
) -> None:
    state_dict = {prefix + "gate.weight": safetensors_weights_manager.get_tensor(prefix + "gate.weight").T}

    assert not config.add_bias

    state_dict.update(
        _get_mlp_weights(
            config=config,
            safetensors_weights_manager=safetensors_weights_manager,
            prefix=prefix,
            column_parallel_shard_dim=column_parallel_shard_dim,
            row_parallel_shard_dim=row_parallel_shard_dim,
        )
    )

    return state_dict
