import torch
from transformers import AutoModelForCausalLM

from ...enums import AttentionHeadType
from ...modeling_utils import split_query_key_value_tensor_for_attention
from ..gpt_dolomite import GPTDolomiteConfig, GPTDolomiteForCausalLM
from .config import GPTCrossLayerConfig
from .main import GPTCrossLayerForCausalLM


def convert_gpt_dolomite_to_gpt_crosslayer(
    original_config: GPTDolomiteConfig,
    original_model: GPTDolomiteForCausalLM,
    sharing_pattern: list[int] | None = None,
    **kwargs,
) -> GPTCrossLayerForCausalLM:
    config = GPTCrossLayerConfig(
        vocab_size=original_config.vocab_size,
        n_positions=original_config.n_positions,
        n_embd=original_config.n_embd,
        n_layer=original_config.n_layer,
        n_head=original_config.n_head,
        num_key_value_heads=original_config.num_key_value_heads,
        n_inner=original_config.n_inner,
        activation_function=original_config.activation_function,
        resid_pdrop=original_config.resid_pdrop,
        embd_pdrop=original_config.embd_pdrop,
        normalization_function=original_config.normalization_function,
        layer_norm_epsilon=original_config.layer_norm_epsilon,
        scale_attn_weights=original_config.scale_attn_weights,
        add_bias=original_config.add_bias,
        position_embedding_type=original_config.position_embedding_type,
        rope_theta=original_config.rope_theta,
        use_cache=original_config.use_cache,
        sharing_pattern=sharing_pattern,
        tie_word_embeddings=original_config.tie_word_embeddings,
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=original_model.dtype, **kwargs)

    attention_head_type = AttentionHeadType(original_config.attention_head_type)
    hidden_size = config.n_embd
    num_attention_heads = config.n_head
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    state_dict = original_model.state_dict()

    sharing_pattern = config.sharing_pattern

    new_state_dict = {"transformer.wte.weight": state_dict["transformer.wte.weight"]}

    if "transformer.wpe.weight" in state_dict:
        new_state_dict["transformer.wpe.weight"] = state_dict["transformer.wpe.weight"]

    if "lm_head.weight" in state_dict:
        new_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]

    new_state_dict["transformer.ln_f.weight"] = state_dict["transformer.ln_f.weight"]
    if config.normalization_function == "layernorm":
        new_state_dict["transformer.ln_f.bias"] = state_dict["transformer.ln_f.bias"]

    for layer_idx in range(original_config.n_layer):
        global_idx, local_idx = model.get_global_local_idx(layer_idx)

        q_attn_weight, k_attn_weight, v_attn_weight = split_query_key_value_tensor_for_attention(
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"],
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )
        if config.add_bias:
            q_attn_bias, k_attn_bias, v_attn_bias = split_query_key_value_tensor_for_attention(
                state_dict.pop(f"transformer.h.{layer_idx}.attn.c_attn.bias"),
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                attention_head_type,
            )

        new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.attn.q_attn.weight"] = q_attn_weight
        if config.add_bias:
            new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.attn.q_attn.bias"] = q_attn_bias

        if layer_idx in sharing_pattern:
            new_state_dict[f"transformer.h.{global_idx}.kv_proj.kv_attn.weight"] = torch.cat(
                [
                    k_attn_weight.view(num_key_value_heads, -1, hidden_size),
                    v_attn_weight.view(num_key_value_heads, -1, hidden_size),
                ],
                dim=1,
            ).view(-1, hidden_size)
            if config.add_bias:
                new_state_dict[f"transformer.h.{global_idx}.kv_proj.kv_attn.bias"] = torch.cat(
                    [k_attn_bias.view(num_key_value_heads, -1), v_attn_bias.view(num_key_value_heads, -1)], dim=1
                ).view(-1)

            new_state_dict[f"transformer.h.{global_idx}.kv_proj.ln.weight"] = state_dict[
                f"transformer.h.{layer_idx}.ln_1.weight"
            ]
            if config.normalization_function == "layernorm":
                new_state_dict[f"transformer.h.{global_idx}.kv_proj.ln.bias"] = state_dict[
                    f"transformer.h.{layer_idx}.ln_1.bias"
                ]

        for ln in ["ln_1", "ln_2"]:
            new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.{ln}.weight"] = state_dict.pop(
                f"transformer.h.{layer_idx}.{ln}.weight"
            )
            if config.normalization_function == "layernorm":
                new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.{ln}.bias"] = state_dict.pop(
                    f"transformer.h.{layer_idx}.{ln}.bias"
                )

        new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.attn.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.attn.c_proj.weight"
        )
        if config.add_bias:
            new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.attn.c_proj.bias"] = state_dict.pop(
                f"transformer.h.{layer_idx}.attn.c_proj.bias"
            )

        new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.mlp.c_fc.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp.c_fc.weight"
        )
        if config.add_bias:
            new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.mlp.c_fc.bias"] = state_dict.pop(
                f"transformer.h.{layer_idx}.mlp.c_fc.bias"
            )

        new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.mlp.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp.c_proj.weight"
        )
        if config.add_bias:
            new_state_dict[f"transformer.h.{global_idx}.layers.{local_idx}.mlp.c_proj.bias"] = state_dict.pop(
                f"transformer.h.{layer_idx}.mlp.c_proj.bias"
            )

    model.load_state_dict(new_state_dict)

    return config, model
