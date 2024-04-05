from typing import Type, Union

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .models import (
    DenseMoEConfig,
    DenseMoEForCausalLM,
    DenseMoEModel,
    GPTMegatronConfig,
    GPTMegatronForCausalLM,
    GPTMegatronModel,
    GPTMultiLayerConfig,
    GPTMultiLayerForCausalLM,
    GPTMultiLayerModel,
    MoEMegablocksConfig,
    MoEMegablocksForCausalLM,
    MoEMegablocksModel,
)


# (AutoConfig, AutoModel, AutoModelForCausalLM)
_CUSTOM_MODEL_REGISTRY = [
    (GPTMegatronConfig, GPTMegatronModel, GPTMegatronForCausalLM),
    (MoEMegablocksConfig, MoEMegablocksModel, MoEMegablocksForCausalLM),
    (GPTMultiLayerConfig, GPTMultiLayerModel, GPTMultiLayerForCausalLM),
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


def is_padding_free_transformer_supported(
    model_class: Union[Type[AutoModelForCausalLM], Type[AutoModelForSeq2SeqLM]], model_type: str
) -> bool:
    return model_class.__name__ in _CUSTOM_MODEL_CLASSES or model_type in _CUSTOM_MODEL_TYPES
