import torch
from tqdm import trange

from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import is_glu
from ...gpt_dolomite import GPTDolomiteConfig


def unshard_gpt_dolomite_tensor_parallel_state_dicts(
    config: GPTDolomiteConfig,
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
            _get_mlp(
                tensor_parallel_state_dicts,
                is_glu=is_glu(config.activation_function),
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
                tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
                prefix=prefix + "lm_head.weight",
                vocab_size=config.vocab_size,
                check_correctness=check_correctness,
            )
        )

    return output_state_dict


def fix_gpt_dolomite_unsharded_state_dict(
    config: GPTDolomiteConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str = ""
) -> dict:
    state_dict[prefix + "transformer.wte.weight"] = state_dict[prefix + "transformer.wte.weight"][
        : config.vocab_size, :
    ]
    state_dict = _fix_attention(config, state_dict, prefix)
    state_dict = _fix_mlp(config, state_dict, tensor_parallel_world_size, prefix)
    return state_dict


def _get_embeddings_or_lm_head(
    tensor_parallel_state_dicts: list[dict],
    tensor_parallel_word_embeddings: bool,
    prefix: str,
    vocab_size: int,
    check_correctness: bool,
) -> dict:
    if tensor_parallel_word_embeddings:
        output = _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts, key=prefix, dim=0)
    else:
        output = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix, check_correctness=check_correctness
        )

    # tensor parallel embeddings uses Embedding_TP class so we need to trim the matrix
    if tensor_parallel_word_embeddings:
        assert output.shape[0] >= vocab_size
        output = output[:vocab_size, :]

    return {prefix: output}


def _get_layernorm(
    tensor_parallel_state_dicts: list[dict], prefix: str, normalization_function: str, check_correctness: bool
) -> dict:
    output = {
        prefix
        + "weight": _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "weight", check_correctness=check_correctness
        )
    }
    if normalization_function == "layernorm":
        output[prefix + "bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "bias", check_correctness=check_correctness
        )
    return output


def _get_attention(
    tensor_parallel_state_dicts: list[dict],
    attention_head_type: AttentionHeadType,
    add_bias: bool,
    prefix: str,
    check_correctness: bool,
) -> dict:
    output = {
        prefix
        + "c_proj.weight": _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.weight", dim=1
        )
    }
    if add_bias:
        output[prefix + "c_proj.bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "c_proj.bias", check_correctness=check_correctness
        )

    if attention_head_type in [AttentionHeadType.mha, AttentionHeadType.gqa]:
        output[prefix + "c_attn.weight"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_attn.weight", dim=0
        )
        if add_bias:
            output[prefix + "c_attn.bias"] = _concatenate_tensors_from_state_dicts(
                tensor_parallel_state_dicts, key=prefix + "c_attn.bias", dim=0
            )
    elif attention_head_type == AttentionHeadType.mqa:
        q_weight = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_attn.q_attn.weight", dim=0
        )
        kv_weight = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "c_attn.kv_attn.weight", check_correctness=check_correctness
        )
        output[prefix + "c_attn.weight"] = torch.cat([q_weight, kv_weight])
        if add_bias:
            q_bias = _concatenate_tensors_from_state_dicts(
                tensor_parallel_state_dicts, key=prefix + "c_attn.q_attn.bias", dim=0
            )
            kv_bias = _get_once_from_state_dicts_with_check(
                tensor_parallel_state_dicts, key=prefix + "c_attn.kv_attn.bias", check_correctness=check_correctness
            )
            output[prefix + "c_attn.bias"] = torch.cat([q_bias, kv_bias])
    else:
        raise ValueError(f"unexpected attention_head_type ({attention_head_type})")

    return output


def _get_mlp(
    tensor_parallel_state_dicts: list[dict], is_glu: bool, add_bias: bool, prefix: str, check_correctness: bool
) -> dict:
    output = {
        prefix
        + "c_proj.weight": _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.weight", dim=1
        )
    }
    if add_bias:
        output[prefix + "c_proj.bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "c_proj.bias", check_correctness=check_correctness
        )

    if is_glu:
        weights = [state_dict[prefix + "c_fc.weight"].chunk(2) for state_dict in tensor_parallel_state_dicts]
        weights = (torch.cat([w[0] for w in weights]), torch.cat([w[1] for w in weights]))
        output[prefix + "c_fc.weight"] = torch.cat(weights)
        if add_bias:
            bias = [state_dict[prefix + "c_fc.bias"].chunk(2) for state_dict in tensor_parallel_state_dicts]
            bias = (torch.cat([b[0] for b in bias]), torch.cat([b[1] for b in bias]))
            output[prefix + "c_fc.bias"] = torch.cat(bias)
    else:
        output[prefix + "c_fc.weight"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_fc.weight", dim=0
        )
        if add_bias:
            output[prefix + "c_fc.bias"] = _concatenate_tensors_from_state_dicts(
                tensor_parallel_state_dicts, key=prefix + "c_fc.bias", dim=0
            )

    return output


def _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts: list[dict], key: str, dim: int) -> torch.Tensor:
    tensor_list = [state_dict[key] for state_dict in tensor_parallel_state_dicts]
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor


def _get_once_from_state_dicts_with_check(
    tensor_parallel_state_dicts: list[dict], key: str, check_correctness: bool
) -> torch.Tensor:
    output: torch.Tensor = tensor_parallel_state_dicts[0][key]
    if check_correctness:
        for state_dict in tensor_parallel_state_dicts[1:]:
            assert output.equal(state_dict[key])
    return output


def _fix_attention(config: GPTDolomiteConfig, state_dict: dict, prefix: str) -> dict:
    if AttentionHeadType(config.attention_head_type) == AttentionHeadType.mqa:
        for layer_idx in range(config.n_layer):
            q_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.attn.c_attn.q_attn.weight")
            kv_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.attn.c_attn.kv_attn.weight")
            state_dict[f"{prefix}transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.cat([q_attn_w, kv_attn_w])

            if config.add_bias:
                q_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.attn.c_attn.q_attn.bias")
                kv_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.attn.c_attn.kv_attn.bias")
                state_dict[f"{prefix}transformer.h.{layer_idx}.attn.c_attn.bias"] = torch.cat([q_attn_w, kv_attn_w])

    return state_dict


def _fix_mlp(config: GPTDolomiteConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str) -> dict:
    if is_glu(config.activation_function):
        for layer_idx in range(config.n_layer):
            key = f"{prefix}transformer.h.{layer_idx}.mlp.c_fc.weight"
            weight = state_dict[key].chunk(tensor_parallel_world_size)
            weight = [w.chunk(2) for w in weight]
            w0 = torch.cat([w[0] for w in weight])
            w1 = torch.cat([w[1] for w in weight])
            state_dict[key] = torch.cat([w0, w1])

            if config.add_bias:
                key = f"{prefix}transformer.h.{layer_idx}.mlp.c_fc.bias"
                weight = state_dict[key].chunk(tensor_parallel_world_size)
                weight = [w.chunk(2) for w in weight]
                w0 = torch.cat([w[0] for w in weight])
                w1 = torch.cat([w[1] for w in weight])
                state_dict[key] = torch.cat([w0, w1])

    return state_dict
