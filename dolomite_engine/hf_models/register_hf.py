from typing import Type, Union

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .models import (
    DenseMoEConfig,
    DenseMoEForCausalLM,
    DenseMoEModel,
    GPTCrossLayerConfig,
    GPTCrossLayerForCausalLM,
    GPTCrossLayerModel,
    GPTDolomiteConfig,
    GPTDolomiteForCausalLM,
    GPTDolomiteModel,
    MoEDolomiteConfig,
    MoEDolomiteForCausalLM,
    MoEDolomiteModel,
)


# (AutoConfig, AutoModel, AutoModelForCausalLM)
_CUSTOM_MODEL_REGISTRY = [
    (GPTDolomiteConfig, GPTDolomiteModel, GPTDolomiteForCausalLM),
    (MoEDolomiteConfig, MoEDolomiteModel, MoEDolomiteForCausalLM),
    (GPTCrossLayerConfig, GPTCrossLayerModel, GPTCrossLayerForCausalLM),
    (DenseMoEConfig, DenseMoEModel, DenseMoEForCausalLM),
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


def is_custom_model(
    model_class: Union[Type[AutoModelForCausalLM], Type[AutoModelForSeq2SeqLM]], model_type: str
) -> bool:
    return model_class.__name__ in _CUSTOM_MODEL_CLASSES or model_type in _CUSTOM_MODEL_TYPES
