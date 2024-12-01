from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .models import (
    DesyncResidualConfig,
    DesyncResidualForCausalLM,
    DesyncResidualModel,
    GPTCrossLayerConfig,
    GPTCrossLayerForCausalLM,
    GPTCrossLayerModel,
    GPTDolomiteConfig,
    GPTDolomiteForCausalLM,
    GPTDolomiteForCausalLM_TP,
    GPTDolomiteModel,
    LadderResidualConfig,
    LadderResidualForCausalLM,
    LadderResidualModel,
    MoEDolomiteConfig,
    MoEDolomiteForCausalLM,
    MoEDolomiteForCausalLM_TP,
    MoEDolomiteModel,
    RNNDolomiteConfig,
    RNNDolomiteForCausalLM,
    RNNDolomiteModel,
)


# (AutoConfig, AutoModel, AutoModelForCausalLM)
_CUSTOM_MODEL_REGISTRY = [
    (GPTDolomiteConfig, GPTDolomiteModel, GPTDolomiteForCausalLM),
    (MoEDolomiteConfig, MoEDolomiteModel, MoEDolomiteForCausalLM),
    (GPTCrossLayerConfig, GPTCrossLayerModel, GPTCrossLayerForCausalLM),
    (RNNDolomiteConfig, RNNDolomiteModel, RNNDolomiteForCausalLM),
    (DesyncResidualConfig, DesyncResidualModel, DesyncResidualForCausalLM),
    (LadderResidualConfig, LadderResidualModel, LadderResidualForCausalLM),
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


def is_custom_model(model_type: str) -> bool:
    return model_type in _CUSTOM_MODEL_TYPES


_MODEL_PARALLEL_CLASS_MAPPING = {
    GPTDolomiteConfig.model_type: GPTDolomiteForCausalLM_TP,
    MoEDolomiteConfig.model_type: MoEDolomiteForCausalLM_TP,
}


def get_model_parallel_class(model_type: str) -> AutoModelForCausalLM:
    if model_type in _MODEL_PARALLEL_CLASS_MAPPING:
        return _MODEL_PARALLEL_CLASS_MAPPING[model_type]

    raise ValueError(f"model parallelism is not supported with `model_type` ({model_type})")
