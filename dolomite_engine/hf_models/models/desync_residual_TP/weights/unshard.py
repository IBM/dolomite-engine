import torch
from tqdm import trange

from ....enums import AttentionHeadType, PositionEmbeddingType
from ...desync_residual import DesyncResidualConfig
from ...gpt_dolomite_TP.weights.unshard import (
    _concatenate_tensors_from_state_dicts,
    _get_embeddings_or_lm_head,
    _get_layernorm,
)


def unshard_desync_residual_tensor_parallel_state_dicts(
    config: DesyncResidualConfig,
    tensor_parallel_state_dicts: list[dict],
    prefix: str = "",
    check_correctness: bool = True,
) -> dict:
    attention_head_type = AttentionHeadType(config.attention_head_type)
    position_embedding_type = PositionEmbeddingType(config.position_embedding_type)

    # word embeddings
    output_state_dict = _get_embeddings_or_lm_head(
        tensor_parallel_state_dicts,
        prefix=prefix + "transformer.wte.weight",
        vocab_size=config.vocab_size,
        check_correctness=check_correctness,
    )

    # positional embeddings if using learned positional embeddings
    if position_embedding_type == PositionEmbeddingType.learned_absolute:
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts,
                prefix=prefix + "transformer.wpe.weight",
                vocab_size=config.n_positions,
                check_correctness=check_correctness,
            )
        )

    # layers
    for layer_idx in trange(config.n_layer):
        # first layernorm
        if layer_idx == 0 or config.reduce_pattern[layer_idx - 1]["mlp"]:
            output_state_dict.update(
                _get_layernorm(
                    tensor_parallel_state_dicts,
                    prefix=prefix + f"transformer.h.{layer_idx}.ln_1.",
                    normalization_function=config.normalization_function,
                    check_correctness=check_correctness,
                )
            )
        else:
            output_state_dict.update(
                _get_desync_residual_layernorm(
                    tensor_parallel_state_dicts,
                    prefix=prefix + f"transformer.h.{layer_idx}.ln_1.",
                    normalization_function=config.normalization_function,
                )
            )

        # attention
        output_state_dict.update(
            _get_attention(
                tensor_parallel_state_dicts,
                attention_head_type=attention_head_type,
                add_bias=config.add_bias,
                prefix=prefix + f"transformer.h.{layer_idx}.attn.",
            )
        )

        # second layernorm
        if config.reduce_pattern[layer_idx]["attention"]:
            output_state_dict.update(
                _get_layernorm(
                    tensor_parallel_state_dicts,
                    prefix=prefix + f"transformer.h.{layer_idx}.ln_2.",
                    normalization_function=config.normalization_function,
                    check_correctness=check_correctness,
                )
            )
        else:
            output_state_dict.update(
                _get_desync_residual_layernorm(
                    tensor_parallel_state_dicts,
                    prefix=prefix + f"transformer.h.{layer_idx}.ln_2.",
                    normalization_function=config.normalization_function,
                )
            )

        # mlp
        output_state_dict.update(
            _get_mlp(
                tensor_parallel_state_dicts,
                add_bias=config.add_bias,
                prefix=prefix + f"transformer.h.{layer_idx}.mlp.",
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
                prefix=prefix + "lm_head.weight",
                vocab_size=config.vocab_size,
                check_correctness=check_correctness,
            )
        )

    return output_state_dict


def fix_desync_residual_unsharded_state_dict(
    config: DesyncResidualConfig, state_dict: dict, tensor_parallel_size: int, prefix: str = ""
) -> dict:
    raise NotImplementedError()


def _get_desync_residual_layernorm(
    tensor_parallel_state_dicts: list[dict], prefix: str, normalization_function: str
) -> dict:
    assert normalization_function == "rmsnorm"

    output = {
        prefix
        + "weight": _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts, key=prefix + "weight", dim=0)
    }

    return output


def _get_attention(
    tensor_parallel_state_dicts: list[dict],
    attention_head_type: AttentionHeadType,
    add_bias: bool,
    prefix: str,
) -> dict:
    output = {
        prefix
        + "c_proj.weight": _concatenate_transposed_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.weight", dim=0
        )
    }
    if add_bias:
        output[prefix + "c_proj.bias"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.bias", dim=0
        )

    if attention_head_type == AttentionHeadType.mqa:
        raise ValueError("DesyncResidual doesn't support mqa")
    else:
        output[prefix + "c_attn.weight"] = _concatenate_transposed_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_attn.weight", dim=0
        )
        if add_bias:
            output[prefix + "c_attn.bias"] = _concatenate_tensors_from_state_dicts(
                tensor_parallel_state_dicts, key=prefix + "c_attn.bias", dim=0
            )

    return output


def _get_mlp(tensor_parallel_state_dicts: list[dict], add_bias: bool, prefix: str, check_correctness: bool) -> dict:
    output = {
        prefix
        + "c_proj.weight": _concatenate_transposed_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.weight", dim=0
        )
    }
    if add_bias:
        output[prefix + "c_proj.bias"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.bias", dim=0
        )

    output[prefix + "c_fc.weight"] = _concatenate_transposed_tensors_from_state_dicts(
        tensor_parallel_state_dicts, key=prefix + "c_fc.weight", dim=0
    )
    if add_bias:
        output[prefix + "c_fc.bias"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_fc.bias", dim=0
        )

    return output


def _concatenate_transposed_tensors_from_state_dicts(
    tensor_parallel_state_dicts: list[dict], key: str, dim: int
) -> torch.Tensor:
    tensor_list = [state_dict[key] for state_dict in tensor_parallel_state_dicts]
    tensor_list = [i.T for i in tensor_list]
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor
