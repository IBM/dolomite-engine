import torch

from .....utils import ProcessGroupManager, SafeTensorsWeightsManager, divide_if_divisible
from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import is_glu
from ....modeling_utils_TP import get_tensor_parallel_vocab_info, tensor_parallel_split_safetensor_slice
from ...gpt_dolomite import GPTDolomiteConfig


def get_gpt_dolomite_model_parallel_state_dict(
    config: GPTDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    tensor_parallel_word_embeddings: bool,
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
                safetensors_weights_manager,
                prefix="transformer.wte.",
                vocab_size=config.vocab_size,
                tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
            )
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
            _get_mlp(
                config=config,
                safetensors_weights_manager=safetensors_weights_manager,
                prefix=prefix + "mlp.",
                column_parallel_shard_dim=0,
                row_parallel_shard_dim=1,
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
                    tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
                )
            )

    return state_dict


def _get_embeddings_or_lm_head(
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
    vocab_size: int,
    tensor_parallel_word_embeddings: bool,
) -> dict:
    if tensor_parallel_word_embeddings:
        vocab_start_index, vocab_end_index, vocab_size_per_tensor_parallel_rank = get_tensor_parallel_vocab_info(
            vocab_size
        )

        weight = safetensors_weights_manager.get_slice(prefix + "weight")[vocab_start_index:vocab_end_index, :]

        if weight.shape[0] < vocab_size_per_tensor_parallel_rank:
            weight = torch.cat(
                [
                    weight,
                    torch.zeros((vocab_size_per_tensor_parallel_rank - weight.shape[0], weight.shape[1])),
                ]
            )
    else:
        weight = safetensors_weights_manager.get_tensor(prefix + "weight")

    return {prefix + "weight": weight}


def _get_layernorm(safetensors_weights_manager: SafeTensorsWeightsManager, prefix: str) -> dict:
    state_dict = {prefix + "weight": safetensors_weights_manager.get_tensor(prefix + "weight")}
    if safetensors_weights_manager.has_tensor(prefix + "bias"):
        state_dict.update({prefix + "bias": safetensors_weights_manager.get_tensor(prefix + "bias")})

    return state_dict


def _get_attention(
    config: GPTDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
    column_parallel_shard_dim: int,
    row_parallel_shard_dim: int,
) -> None:
    state_dict = {}

    if AttentionHeadType(config.attention_head_type) == AttentionHeadType.mqa:
        tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        global_hidden_size = config.n_embd
        head_dim = divide_if_divisible(global_hidden_size, config.n_head, "")

        hidden_size_per_rank = divide_if_divisible(global_hidden_size, tp_world_size, "")
        start_index = tp_rank * hidden_size_per_rank
        end_index = (tp_rank + 1) * hidden_size_per_rank

        weight = safetensors_weights_manager.get_slice(prefix + "c_attn.weight")
        state_dict[prefix + "c_attn.q_attn.weight"] = weight[start_index:end_index, :]
        state_dict[prefix + "c_attn.kv_attn.weight"] = weight[
            global_hidden_size : global_hidden_size + 2 * head_dim, :
        ]

        if config.add_bias:
            bias = safetensors_weights_manager.get_slice(prefix + "c_attn.bias")
            state_dict[prefix + "c_attn.q_attn.bias"] = bias[start_index:end_index]
            state_dict[prefix + "c_attn.kv_attn.bias"] = bias[global_hidden_size : global_hidden_size + 2 * head_dim]
    else:
        state_dict.update(
            _get_column_parallel(
                config=config,
                safetensors_weights_manager=safetensors_weights_manager,
                prefix=prefix + "c_attn.",
                shard_dim=column_parallel_shard_dim,
            )
        )

    state_dict.update(
        _get_row_parallel(
            config=config,
            safetensors_weights_manager=safetensors_weights_manager,
            prefix=prefix + "c_proj.",
            shard_dim=row_parallel_shard_dim,
        )
    )

    return state_dict


def _get_mlp(
    config: GPTDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
    column_parallel_shard_dim: int,
    row_parallel_shard_dim: int,
) -> None:
    # GLU is a special case and needs to be handled explicitely
    if is_glu(config.activation_function):
        weight = safetensors_weights_manager.get_slice(prefix + "c_fc.weight")

        tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        shape = weight.get_shape()
        stride = divide_if_divisible(
            shape[column_parallel_shard_dim],
            tp_world_size * 2,
            f"split dimension ({column_parallel_shard_dim}) is not divisible by 2 x tensor parallel world size (2 x {tp_world_size})",
        )

        # split weight tensors into gate and non-gate
        start_end = (tp_rank * stride, (tp_rank + 1) * stride)
        weight_1 = tensor_parallel_split_safetensor_slice(weight, column_parallel_shard_dim, start_end)
        if config.add_bias:
            bias = safetensors_weights_manager.get_slice(prefix + "c_fc.bias")
            bias_1 = tensor_parallel_split_safetensor_slice(bias, column_parallel_shard_dim, start_end)

        start_end = (
            (tp_world_size + tp_rank) * stride,
            (tp_world_size + tp_rank + 1) * stride,
        )
        weight_2 = tensor_parallel_split_safetensor_slice(weight, column_parallel_shard_dim, start_end)
        if config.add_bias:
            bias_2 = tensor_parallel_split_safetensor_slice(bias, column_parallel_shard_dim, start_end)

        state_dict = {prefix + "c_fc.weight": torch.cat([weight_1, weight_2])}
        if config.add_bias:
            state_dict[prefix + "c_fc.bias"] = torch.cat([bias_1, bias_2])
    else:
        state_dict = _get_column_parallel(
            config=config,
            safetensors_weights_manager=safetensors_weights_manager,
            prefix=prefix + "c_fc.",
            shard_dim=column_parallel_shard_dim,
        )

    state_dict.update(
        _get_row_parallel(
            config=config,
            safetensors_weights_manager=safetensors_weights_manager,
            prefix=prefix + "c_proj.",
            shard_dim=row_parallel_shard_dim,
        )
    )

    return state_dict


def _get_column_parallel(
    config: GPTDolomiteConfig, safetensors_weights_manager: SafeTensorsWeightsManager, prefix: str, shard_dim: int
) -> dict:
    weight = safetensors_weights_manager.get_slice(prefix + "weight")
    weight = tensor_parallel_split_safetensor_slice(weight, dim=shard_dim)
    state_dict = {prefix + "weight": weight}

    if config.add_bias:
        bias = safetensors_weights_manager.get_slice(prefix + "bias")
        bias = tensor_parallel_split_safetensor_slice(bias, dim=shard_dim)
        state_dict[prefix + "bias"] = bias

    return state_dict


def _get_row_parallel(
    config: GPTDolomiteConfig, safetensors_weights_manager: SafeTensorsWeightsManager, prefix: str, shard_dim: int
) -> dict:
    weight = safetensors_weights_manager.get_slice(prefix + "weight")
    weight = tensor_parallel_split_safetensor_slice(weight, dim=shard_dim)
    state_dict = {prefix + "weight": weight}

    if config.add_bias:
        state_dict[prefix + "bias"] = safetensors_weights_manager.get_tensor(prefix + "bias")

    return state_dict
