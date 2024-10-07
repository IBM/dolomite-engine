import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from ...utils import SafeTensorsWeightsManager, download_repo
from ..enums import AttentionHeadType
from ..modeling_utils import (
    interleave_query_key_value_tensor_for_attention,
    split_query_key_value_tensor_for_attention,
)
from ..models import MoEDolomiteConfig


try:
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
except:
    GraniteMoeConfig = None


def import_from_huggingface_granitemoe(pretrained_model_name_or_path: str, save_path: str) -> None:
    original_config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)
    config = _import_config_from_huggingface(original_config)

    safetensors_weights_manager = SafeTensorsWeightsManager(downloaded_model_path)
    state_dict = _import_state_dict_from_huggingface(
        safetensors_weights_manager,
        config.n_layer,
        config.num_experts,
        config.n_head,
        config.num_key_value_heads,
        config.n_embd // config.n_head,
        AttentionHeadType(config.attention_head_type),
    )

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    config.save_pretrained(save_path)

    generation_config = GenerationConfig.from_model_config(config)
    generation_config.save_pretrained(save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path, legacy_format=False)


def _import_config_from_huggingface(original_config: GraniteMoeConfig) -> MoEDolomiteConfig:
    assert original_config.hidden_act == "silu"

    if original_config.num_attention_heads == original_config.num_key_value_heads:
        attention_head_type = "mha"
    elif original_config.num_key_value_heads == 1:
        attention_head_type = "mqa"
    elif original_config.num_attention_heads > original_config.num_key_value_heads:
        attention_head_type = "gqa"

    assert not original_config.attention_bias

    config = MoEDolomiteConfig(
        vocab_size=original_config.vocab_size,
        n_positions=original_config.max_position_embeddings,
        n_embd=original_config.hidden_size,
        n_layer=original_config.num_hidden_layers,
        n_head=original_config.num_attention_heads,
        num_key_value_heads=original_config.num_key_value_heads,
        attention_head_type=attention_head_type,
        position_embedding_type="rope",
        n_inner=original_config.intermediate_size,
        activation_function="swiglu",
        normalization_function="rmsnorm",
        layer_norm_epsilon=original_config.rms_norm_eps,
        use_cache=original_config.use_cache,
        add_bias=original_config.attention_bias,
        tie_word_embeddings=original_config.tie_word_embeddings,
        initializer_range=original_config.initializer_range,
        rope_theta=original_config.rope_theta,
        rope_scaling=original_config.rope_scaling,
        attn_pdrop=original_config.attention_dropout,
        num_experts=original_config.num_local_experts,
        num_experts_per_tok=original_config.num_experts_per_tok,
        output_router_logits=original_config.output_router_logits,
        router_aux_loss_coef=original_config.router_aux_loss_coef,
        bos_token_id=original_config.bos_token_id,
        eos_token_id=original_config.eos_token_id,
        pad_token_id=original_config.pad_token_id,
        m_emb=None if original_config.embedding_multiplier == 1 else original_config.embedding_multiplier,
        m_residual=None if original_config.residual_multiplier == 1 else original_config.residual_multiplier,
        m_width=None if original_config.logits_scaling == 1 else original_config.logits_scaling,
        attention_multiplier=original_config.attention_multiplier,
    )

    return config


def _import_state_dict_from_huggingface(
    safetensors_weights_manager: SafeTensorsWeightsManager,
    num_layers: int,
    num_experts: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_head_type: AttentionHeadType,
) -> None:
    state_dict = {
        "transformer.wte.weight": safetensors_weights_manager.get_tensor("model.embed_tokens.weight"),
        "transformer.ln_f.weight": safetensors_weights_manager.get_tensor("model.norm.weight"),
    }

    if safetensors_weights_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weights_manager.get_tensor("lm_head.weight")

    for layer_idx in range(num_layers):
        state_dict[f"transformer.h.{layer_idx}.ln_1.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.input_layernorm.weight"
        )
        state_dict[f"transformer.h.{layer_idx}.ln_2.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )

        state_dict[f"transformer.h.{layer_idx}.moe.gate.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.block_sparse_moe.router.layer.weight"
        ).T.contiguous()

        state_dict[f"transformer.h.{layer_idx}.moe.c_fc.weight"] = (
            _split_and_reorder_for_glu(
                safetensors_weights_manager.get_tensor(
                    f"model.layers.{layer_idx}.block_sparse_moe.input_linear.weight"
                )
            )
            .transpose(0, 1)
            .contiguous()
        )
        state_dict[f"transformer.h.{layer_idx}.moe.c_proj.weight"] = (
            safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.block_sparse_moe.output_linear.weight")
            .transpose(0, 1)
            .contiguous()
        )

        state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = interleave_query_key_value_tensor_for_attention(
            safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
            safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
            safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
            num_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )
        state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )

    return state_dict


def export_to_huggingface_granitemoe(pretrained_model_name_or_path: str, save_path: str) -> None:
    config: MoEDolomiteConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    original_config = _export_config_to_huggingface(config)

    safetensors_weights_manager = SafeTensorsWeightsManager(pretrained_model_name_or_path)
    state_dict = _export_state_dict_to_huggingface(
        safetensors_weights_manager,
        config.n_layer,
        config.num_experts,
        config.n_head,
        config.num_key_value_heads,
        config.n_embd // config.n_head,
        AttentionHeadType(config.attention_head_type),
    )

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    original_config.save_pretrained(save_path)

    original_generation_config = GenerationConfig.from_model_config(original_config)
    original_generation_config.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        tokenizer.save_pretrained(save_path, legacy_format=False)
    except:
        pass


def _export_config_to_huggingface(config: MoEDolomiteConfig) -> GraniteMoeConfig:
    assert config.activation_function == "swiglu"
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "rope"
    assert not config.add_bias

    original_config = GraniteMoeConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.n_positions,
        hidden_size=config.n_embd,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=4 * config.n_embd if config.n_inner is None else config.n_inner,
        hidden_act="silu",
        rms_norm_eps=config.layer_norm_epsilon,
        use_cache=config.use_cache,
        attention_bias=config.add_bias,
        tie_word_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        attention_dropout=config.attn_pdrop,
        num_local_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        output_router_logits=config.output_router_logits,
        router_aux_loss_coef=config.router_aux_loss_coef,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        embedding_multiplier=1 if config.m_emb is None else config.m_emb,
        residual_multiplier=1 if config.m_residual is None else config.m_residual,
        logits_scaling=1 if config.m_width is None else config.m_width,
        attention_multiplier=config.attention_multiplier,
        architectures=[GraniteMoeForCausalLM.__name__],
    )

    return original_config


def _export_state_dict_to_huggingface(
    safetensors_weights_manager: SafeTensorsWeightsManager,
    num_layers: int,
    num_experts: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_head_type: AttentionHeadType,
) -> None:
    state_dict = {
        "model.embed_tokens.weight": safetensors_weights_manager.get_tensor("transformer.wte.weight"),
        "model.norm.weight": safetensors_weights_manager.get_tensor("transformer.ln_f.weight"),
    }

    if safetensors_weights_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weights_manager.get_tensor("lm_head.weight")

    for layer_idx in range(num_layers):
        state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.ln_1.weight"
        )
        state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = (
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.ln_2.weight")
        )

        state_dict[f"model.layers.{layer_idx}.block_sparse_moe.router.layer.weight"] = (
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.moe.gate.weight")
        ).T.contiguous()

        state_dict[f"model.layers.{layer_idx}.block_sparse_moe.input_linear.weight"] = _split_and_reorder_for_glu(
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.moe.c_fc.weight").transpose(0, 1)
        ).contiguous()
        state_dict[f"model.layers.{layer_idx}.block_sparse_moe.output_linear.weight"] = (
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.moe.c_proj.weight").transpose(0, 1)
        ).contiguous()

        query_weight, key_weight, value_weight = split_query_key_value_tensor_for_attention(
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.attn.c_attn.weight"),
            num_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )
        state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = query_weight
        state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = key_weight
        state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = value_weight

        state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.attn.c_proj.weight"
        )

    return state_dict


def _split_and_reorder_for_glu(weight: torch.Tensor) -> torch.Tensor:
    x, y = weight.chunk(2, dim=1)
    weight = torch.cat([y, x], dim=1)
    return weight
