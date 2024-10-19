from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .models import (
    GPTCrossLayerConfig,
    GPTCrossLayerForCausalLM,
    GPTCrossLayerModel,
    GPTDolomiteConfig,
    GPTDolomiteForCausalLM,
    GPTDolomiteForCausalLM_TP,
    GPTDolomiteModel,
    MoEDolomiteConfig,
    MoEDolomiteForCausalLM,
    MoEDolomiteForCausalLM_TP,
    MoEDolomiteModel,
    RNNDolomiteConfig,
    RNNDolomiteForCausalLM,
    RNNDolomiteModel,
    SBDolomiteConfig,
    SBDolomiteForCausalLM,
    SBDolomiteModel,
)


# (AutoConfig, AutoModel, AutoModelForCausalLM)
_CUSTOM_MODEL_REGISTRY = [
    (GPTDolomiteConfig, GPTDolomiteModel, GPTDolomiteForCausalLM),
    (MoEDolomiteConfig, MoEDolomiteModel, MoEDolomiteForCausalLM),
    (GPTCrossLayerConfig, GPTCrossLayerModel, GPTCrossLayerForCausalLM),
    (RNNDolomiteConfig, RNNDolomiteModel, RNNDolomiteForCausalLM),
    (SBDolomiteConfig, SBDolomiteModel, SBDolomiteForCausalLM),
]
_CUSTOM_MODEL_TYPES = []
_CUSTOM_MODEL_CLASSES = []


def register_model_classes() -> None:
    for config_class, auto_model_class, auto_model_for_causal_lm_class in _CUSTOM_MODEL_REGISTRY:
        model_type = config_class.model_type

        AutoConfig.register(model_type, config_class)
        AutoModel.register(config_class, auto_model_class)
        AutoModelForCausalLM.register(config_class, auto_model_for_causal_lm_class)

        _CUSTOM_MODEL_TYPES.append(model_type)
        _CUSTOM_MODEL_CLASSES.append(auto_model_for_causal_lm_class)


def is_custom_model(model_class: type[AutoModelForCausalLM] | type[AutoModelForSeq2SeqLM], model_type: str) -> bool:
    return model_class.__name__ in _CUSTOM_MODEL_CLASSES or model_type in _CUSTOM_MODEL_TYPES


_TENSOR_PARALLEL_CLASS_MAPPING = {
    GPTDolomiteConfig.model_type: GPTDolomiteForCausalLM_TP,
    MoEDolomiteConfig.model_type: MoEDolomiteForCausalLM_TP,
}


def get_tensor_parallel_class(model_type: str) -> AutoModelForCausalLM:
    if model_type in _TENSOR_PARALLEL_CLASS_MAPPING:
        return _TENSOR_PARALLEL_CLASS_MAPPING[model_type]

    raise ValueError(f"tensor parallel is not supported with `model_type` ({model_type})")
