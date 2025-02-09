from transformers import AutoConfig, AutoTokenizer, GenerationConfig, GraniteConfig, GraniteForCausalLM

from ...utils import SafeTensorsWeightsManager, download_repo
from ..enums import AttentionHeadType
from ..models import GPTDolomiteConfig
from .llama import _export_state_dict_to_huggingface, _import_state_dict_from_huggingface


def import_from_huggingface_granite(pretrained_model_name_or_path: str, save_path: str) -> None:
    original_config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)
    config = _import_config_from_huggingface(original_config)

    safetensors_weights_manager = SafeTensorsWeightsManager(downloaded_model_path)
    state_dict = _import_state_dict_from_huggingface(
        safetensors_weights_manager,
        config.num_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
        AttentionHeadType(config.attention_head_type),
    )

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    config.save_pretrained(save_path)

    generation_config = GenerationConfig.from_model_config(config)
    generation_config.save_pretrained(save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path, legacy_format=False)


def _import_config_from_huggingface(original_config: GraniteConfig) -> GPTDolomiteConfig:
    assert original_config.hidden_act == "silu"

    if original_config.num_attention_heads == original_config.num_key_value_heads:
        attention_head_type = "mha"
    elif original_config.num_key_value_heads == 1:
        attention_head_type = "mqa"
    elif original_config.num_attention_heads > original_config.num_key_value_heads:
        attention_head_type = "gqa"

    assert original_config.mlp_bias == original_config.attention_bias

    config = GPTDolomiteConfig(
        vocab_size=original_config.vocab_size,
        max_position_embeddings=original_config.max_position_embeddings,
        hidden_size=original_config.hidden_size,
        num_layers=original_config.num_hidden_layers,
        num_attention_heads=original_config.num_attention_heads,
        num_key_value_heads=original_config.num_key_value_heads,
        attention_head_type=attention_head_type,
        position_embedding_type="rope",
        normalization_function="rmsnorm",
        layer_norm_epsilon=original_config.rms_norm_eps,
        use_cache=original_config.use_cache,
        add_bias=original_config.attention_bias,
        tie_word_embeddings=original_config.tie_word_embeddings,
        initializer_range=original_config.initializer_range,
        rope_theta=original_config.rope_theta,
        rope_scaling=original_config.rope_scaling,
        attn_pdrop=original_config.attention_dropout,
        bos_token_id=original_config.bos_token_id,
        eos_token_id=original_config.eos_token_id,
        pad_token_id=original_config.pad_token_id,
        m_emb=None if original_config.embedding_multiplier == 1 else original_config.embedding_multiplier,
        m_residual=None if original_config.residual_multiplier == 1 else original_config.residual_multiplier,
        m_width=None if original_config.logits_scaling == 1 else original_config.logits_scaling,
        attention_multiplier=original_config.attention_multiplier,
        mlp_blocks=[
            {
                "mlp_block_type": "MLP",
                "activation_function": "swiglu",
                "intermediate_size": original_config.intermediate_size,
            }
            for _ in range(original_config.num_hidden_layers)
        ],
    )

    return config


def export_to_huggingface_granite(pretrained_model_name_or_path: str, save_path: str) -> None:
    config: GPTDolomiteConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    original_config = _export_config_to_huggingface(config)

    safetensors_weights_manager = SafeTensorsWeightsManager(pretrained_model_name_or_path)
    state_dict = _export_state_dict_to_huggingface(
        safetensors_weights_manager,
        config.num_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
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


def _export_config_to_huggingface(config: GPTDolomiteConfig) -> GraniteConfig:
    assert config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function") == "swiglu"
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "rope"
    config.check_equal_for_all_and_get_value("mlp_blocks", "mlp_block_type")

    original_config = GraniteConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=config.check_equal_for_all_and_get_value("mlp_blocks", "intermediate_size"),
        hidden_act="silu",
        rms_norm_eps=config.layer_norm_epsilon,
        use_cache=config.use_cache,
        attention_bias=config.add_bias,
        tie_word_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        attention_dropout=config.attn_pdrop,
        mlp_bias=config.add_bias,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        embedding_multiplier=1 if config.m_emb is None else config.m_emb,
        residual_multiplier=1 if config.m_residual is None else config.m_residual,
        logits_scaling=1 if config.m_width is None else config.m_width,
        attention_multiplier=config.attention_multiplier,
        architectures=[GraniteForCausalLM.__name__],
    )

    return original_config
