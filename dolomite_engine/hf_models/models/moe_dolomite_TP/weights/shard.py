import torch

from .....utils import ProcessGroupManager, SafeTensorsWeightsManager
from ....enums import PositionEmbeddingType
from ....modeling_utils import is_glu
from ....utils import divide_if_divisible
from ...gpt_dolomite_TP.weights.shard import _get_attention, _get_embeddings_or_lm_head, _get_layernorm
from ...moe_dolomite import MoEDolomiteConfig


def get_moe_dolomite_tp_state_dict(
    config: MoEDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    tensor_parallel_word_embeddings: bool,
) -> dict:
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

        state_dict.update(_get_layernorm(safetensors_weights_manager, prefix=prefix + "ln_1."))

        state_dict.update(
            _get_attention(
                config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "attn."
            )
        )

        state_dict.update(_get_layernorm(safetensors_weights_manager, prefix=prefix + "ln_2."))

        state_dict.update(
            _get_moe(config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "moe.")
        )

    state_dict.update(_get_layernorm(safetensors_weights_manager, prefix=prefix + "ln_f."))

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


def _get_moe(
    config: MoEDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
) -> None:
    # GLU is a special case and needs to be handled explicitely
    state_dict = {prefix + "gate.weight": safetensors_weights_manager.get_tensor(prefix + "gate.weight")}
    weight = safetensors_weights_manager.get_tensor(prefix + "c_fc.weight")
    tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
    if is_glu(config.activation_function):
        # weight = safetensors_weights_manager.get_slice(prefix + "c_fc.weight")
        shape = (config.n_inner * 2, config.num_experts, config.n_embd)
        sharded_out_dim = divide_if_divisible(
            shape[0],
            tp_world_size * 2,
            f"split dimension ({0}) is not divisible by 2 x tensor parallel world size (2 x {tp_world_size})",
        )
        weight = weight.view(tp_world_size, sharded_out_dim, config.num_experts, config.n_embd)
        # split weight tensors into gate and non-gate
        weight_1 = weight[tp_rank]
        weight_2 = weight[tp_world_size + tp_rank]
        state_dict[prefix + "c_fc.weight"] = torch.cat([weight_1, weight_2], dim=1)
    else:
        shape = (config.n_inner, config.num_experts, config.n_embd)
        sharded_out_dim = divide_if_divisible(
            shape[0],
            tp_world_size,
            f"split dimension ({0}) is not divisible by tensor parallel world size ({tp_world_size})",
        )
        weight = weight.view(tp_world_size, sharded_out_dim, config.num_experts, config.n_embd)
        # split weight tensors into gate and non-gate
        weight = weight[tp_rank]
        state_dict[prefix + "c_fc.weight"] = weight

    weight = safetensors_weights_manager.get_tensor(prefix + "c_proj.weight")
    sharded_in_dim = sharded_out_dim
    weight = weight.view(config.n_embd, config.num_experts, tp_world_size, sharded_in_dim)
    state_dict[prefix + "c_proj.weight"] = weight[:, :, tp_rank]

    return state_dict
