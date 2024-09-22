from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .models import (
    GPTCrossLayerConfig,
    GPTCrossLayerForCausalLM,
    GPTCrossLayerModel,
    GPTDolomiteConfig,
    GPTDolomiteForCausalLM,
    GPTDolomiteForCausalLM_TP,
    GPTDolomiteModel,
    GPTEnsembleConfig,
    GPTEnsembleForCausalLM,
    GPTEnsembleForCausalLM_TP,
    GPTEnsembleModel,
    GPTLadderConfig,
    GPTLadderForCausalLM,
    GPTLadderModel,
    GPTParallelConfig,
    GPTParallelForCausalLM,
    GPTParallelForCausalLM_TP,
    GPTParallelModel,
    MoEDolomiteConfig,
    MoEDolomiteForCausalLM,
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
    (GPTEnsembleConfig, GPTEnsembleModel, GPTEnsembleForCausalLM),
    (GPTLadderConfig, GPTLadderModel, GPTLadderForCausalLM),
    (GPTParallelConfig, GPTParallelModel, GPTParallelForCausalLM),
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
    "gpt_dolomite": (GPTDolomiteForCausalLM, GPTDolomiteForCausalLM_TP),
    "gpt_ensemble": (GPTEnsembleForCausalLM, GPTEnsembleForCausalLM_TP),
    "gpt_parallel": (GPTParallelForCausalLM, GPTParallelForCausalLM_TP),
}


def get_tensor_parallel_class(model_type: str) -> AutoModelForCausalLM:
    assert model_type in _TENSOR_PARALLEL_CLASS_MAPPING, "tensor parallel is not supported with this model"
    return _TENSOR_PARALLEL_CLASS_MAPPING[model_type][1]
