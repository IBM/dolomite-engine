import torch
from tqdm import trange

from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import is_glu
from ...gpt_dolomite_TP.weights.unshard import (
    _concatenate_tensors_from_state_dicts,
    _fix_attention,
    _get_attention,
    _get_embeddings_or_lm_head,
    _get_layernorm,
    _get_once_from_state_dicts_with_check,
)
from ...moe_dolomite import MoEDolomiteConfig


def unshard_moe_dolomite_tensor_parallel_state_dicts(
    config: MoEDolomiteConfig,
    tensor_parallel_state_dicts: list[dict],
    tensor_parallel_word_embeddings: bool,
    prefix: str = "",
    check_correctness: bool = True,
) -> dict:
    attention_head_type = AttentionHeadType(config.attention_head_type)
    position_embedding_type = PositionEmbeddingType(config.position_embedding_type)

    # word embeddings
    output_state_dict = _get_embeddings_or_lm_head(
        tensor_parallel_state_dicts,
        tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
        prefix=prefix + "transformer.wte.weight",
        vocab_size=config.vocab_size,
        check_correctness=check_correctness,
    )

    # positional embeddings if using learned positional embeddings
    if position_embedding_type == PositionEmbeddingType.learned_absolute:
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts,
                # TODO change this if we support tensor parallel position embeddings
                tensor_parallel_word_embeddings=False,
                prefix=prefix + "transformer.wpe.weight",
                vocab_size=config.n_positions,
                check_correctness=check_correctness,
            )
        )

    # layers
    for layer_idx in trange(config.n_layer):
        # first layernorm
        output_state_dict.update(
            _get_layernorm(
                tensor_parallel_state_dicts,
                prefix=prefix + f"transformer.h.{layer_idx}.ln_1.",
                normalization_function=config.normalization_function,
                check_correctness=check_correctness,
            )
        )

        # attention
        output_state_dict.update(
            _get_attention(
                tensor_parallel_state_dicts,
                attention_head_type=attention_head_type,
                add_bias=config.add_bias,
                prefix=prefix + f"transformer.h.{layer_idx}.attn.",
                check_correctness=check_correctness,
            )
        )

        # second layernorm
        output_state_dict.update(
            _get_layernorm(
                tensor_parallel_state_dicts,
                prefix=prefix + f"transformer.h.{layer_idx}.ln_2.",
                normalization_function=config.normalization_function,
                check_correctness=check_correctness,
            )
        )

        # mlp
        output_state_dict.update(
            _get_moe(
                tensor_parallel_state_dicts,
                prefix=prefix + f"transformer.h.{layer_idx}.moe.",
                config=config,
                check_correctness=check_correctness,
            )
        )

    # final layernorm
    output_state_dict.update(
        _get_layernorm(
            tensor_parallel_state_dicts,
            prefix=prefix + f"transformer.ln_f.",
            normalization_function=config.normalization_function,
            check_correctness=check_correctness,
        )
    )

    if not config.tie_word_embeddings:
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts,
                tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
                prefix=prefix + "lm_head.weight",
                vocab_size=config.vocab_size,
                check_correctness=check_correctness,
            )
        )

    return output_state_dict


def fix_moe_dolomite_unsharded_state_dict(
    config: MoEDolomiteConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str = ""
) -> dict:
    state_dict[prefix + "transformer.wte.weight"] = state_dict[prefix + "transformer.wte.weight"][
        : config.vocab_size, :
    ]
    state_dict = _fix_attention(config, state_dict, prefix)
    state_dict = _fix_moe(config, state_dict, tensor_parallel_world_size, prefix)
    return state_dict


def _concatenate_tensors_from_moe(
    tensor_parallel_state_dicts: list[dict],
    key: str,
    dim: int,
) -> torch.Tensor:
    tensor_list = [state_dict[key] for state_dict in tensor_parallel_state_dicts]
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor


def _get_moe(
    tensor_parallel_state_dicts: list[dict], config: MoEDolomiteConfig, prefix: str, check_correctness: bool
) -> dict:
    assert not config.add_bias

    output = {
        prefix
        + "gate.weight": _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, prefix + "gate.weight", True
        )
    }

    if is_glu(config.activation_function):
        # per_rank_dim = config.n_inner // len(tensor_parallel_state_dicts)
        weights = [state_dict[prefix + "c_fc.weight"].chunk(2, dim=0) for state_dict in tensor_parallel_state_dicts]
        weights = (torch.cat([w[0] for w in weights], dim=0), torch.cat([w[1] for w in weights], dim=0))
        output[prefix + "c_fc.weight"] = torch.cat(weights, dim=0)
    else:
        output[prefix + "c_fc.weight"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, prefix + "c_fc.weight", dim=0
        )

    output[prefix + "c_proj.weight"] = _concatenate_tensors_from_moe(
        tensor_parallel_state_dicts, prefix + "c_proj.weight", dim=2
    )
    return output


def _fix_moe(config: MoEDolomiteConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str) -> dict:
    assert not config.add_bias

    if is_glu(config.activation_function):
        for layer_idx in range(config.n_layer):
            key = f"{prefix}transformer.h.{layer_idx}.moe.c_fc.weight"
            weight = state_dict[key]
            weight = weight.chunk(tensor_parallel_world_size, dim=0)
            weight = [w.chunk(2, dim=0) for w in weight]
            w0 = torch.cat([w[0] for w in weight])
            w1 = torch.cat([w[1] for w in weight])
            state_dict[key] = torch.cat([w0, w1], dim=0)

    return state_dict
