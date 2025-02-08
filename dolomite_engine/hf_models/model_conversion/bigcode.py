from transformers import AutoConfig, AutoTokenizer, GenerationConfig, GPTBigCodeConfig, GPTBigCodeForCausalLM

from ...utils import SafeTensorsWeightsManager, download_repo
from ..enums import AttentionHeadType, PositionEmbeddingType
from ..models import GPTDolomiteConfig


def import_from_huggingface_bigcode(pretrained_model_name_or_path: str, save_path: str) -> None:
    original_config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)
    config = _import_config_from_huggingface(original_config)

    safetensors_weights_manager = SafeTensorsWeightsManager(downloaded_model_path)
    state_dict = _import_state_dict_from_huggingface(safetensors_weights_manager, config.num_layers)

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    config.save_pretrained(save_path)

    generation_config = GenerationConfig.from_model_config(config)
    generation_config.save_pretrained(save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path, legacy_format=False)


def _import_config_from_huggingface(original_config: GPTBigCodeConfig) -> GPTDolomiteConfig:
    assert original_config.activation_function in ["gelu_pytorch_tanh", "gelu"]

    config = GPTDolomiteConfig(
        vocab_size=original_config.vocab_size,
        max_position_embeddings=original_config.n_positions,
        hidden_size=original_config.n_embd,
        num_layers=original_config.n_layer,
        num_attention_heads=original_config.n_head,
        attention_head_type="mqa" if original_config.multi_query else "mha",
        position_embedding_type="learned_absolute",
        intermediate_size=original_config.n_inner,
        activation_function=original_config.activation_function,
        normalization_function="layernorm",
        layer_norm_epsilon=original_config.layer_norm_epsilon,
        use_cache=original_config.use_cache,
        add_bias=True,
        tie_word_embeddings=original_config.tie_word_embeddings,
        initializer_range=original_config.initializer_range,
        attn_pdrop=original_config.attn_pdrop,
        resid_pdrop=original_config.resid_pdrop,
        bos_token_id=original_config.bos_token_id,
        eos_token_id=original_config.eos_token_id,
        pad_token_id=original_config.pad_token_id,
    )

    return config


def _import_state_dict_from_huggingface(
    safetensors_weights_manager: SafeTensorsWeightsManager, num_layers: int
) -> None:
    state_dict = {key: safetensors_weights_manager.get_tensor(key) for key in safetensors_weights_manager}

    for layer_idx in range(num_layers):
        # sequence_mixer.c_attn
        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.attn.c_attn.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.attn.c_attn.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.bias"] = bias

        # sequence_mixer.c_proj
        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.attn.c_proj.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.attn.c_proj.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias"] = bias

        # mlp_block.c_fc
        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp.c_fc.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.mlp.c_fc.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.bias"] = bias

        # mlp_block.c_proj
        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp.c_proj.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.mlp.c_proj.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.bias"] = bias

    return state_dict


def export_to_huggingface_bigcode(pretrained_model_name_or_path: str, save_path: str) -> None:
    config: GPTDolomiteConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    original_config = _export_config_to_huggingface(config)

    safetensors_weights_manager = SafeTensorsWeightsManager(pretrained_model_name_or_path)
    state_dict = _export_state_dict_to_huggingface(safetensors_weights_manager, config.num_layers)

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    original_config.save_pretrained(save_path)

    original_generation_config = GenerationConfig.from_model_config(original_config)
    original_generation_config.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        tokenizer.save_pretrained(save_path, legacy_format=False)
    except:
        pass


def _export_config_to_huggingface(config: GPTDolomiteConfig) -> GPTBigCodeConfig:
    assert config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function") == "gelu_pytorch_tanh"
    assert config.normalization_function == "layernorm"
    assert AttentionHeadType(config.attention_head_type) in [AttentionHeadType.mha, AttentionHeadType.mqa]
    assert PositionEmbeddingType(config.position_embedding_type) == PositionEmbeddingType.learned_absolute
    assert config.m_emb is None
    assert config.m_residual is None
    assert config.m_width is None
    assert config.attention_multiplier is None

    original_config = GPTBigCodeConfig(
        vocab_size=config.vocab_size,
        n_positions=config.max_position_embeddings,
        n_embd=config.hidden_size,
        n_layer=config.num_layers,
        n_head=config.num_attention_heads,
        n_inner=config.check_equal_for_all_and_get_value("mlp_blocks", "intermediate_size"),
        activation_function=config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function"),
        resid_pdrop=config.resid_pdrop,
        embd_pdrop=config.embd_pdrop,
        attn_pdrop=config.attn_pdrop,
        layer_norm_epsilon=config.layer_norm_epsilon,
        initializer_range=config.initializer_range,
        use_cache=config.use_cache,
        multi_query=config.multi_query,
        tie_word_embeddings=config.tie_word_embeddings,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        architectures=[GPTBigCodeForCausalLM.__name__],
    )

    return original_config


def _export_state_dict_to_huggingface(safetensors_weights_manager: SafeTensorsWeightsManager, num_layers: int) -> dict:
    state_dict = {key: safetensors_weights_manager.get_tensor(key) for key in safetensors_weights_manager}

    for layer_idx in range(num_layers):
        # sequence_mixer.c_attn
        state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.sequence_mixer.c_attn.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.bias"] = bias

        # sequence_mixer.c_proj
        state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.attn.c_proj.bias"] = bias

        # mlp_block.c_fc
        state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.mlp_block.c_fc.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.bias"] = bias

        # mlp_block.c_proj
        state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"
        )

    return state_dict
