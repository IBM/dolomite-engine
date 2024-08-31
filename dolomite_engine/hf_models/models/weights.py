from ...utils import SafeTensorsWeightsManager
from ..config import CommonConfig
from .gpt_dolomite import GPTDolomiteConfig
from .gpt_dolomite_TP import (
    fix_gpt_dolomite_unsharded_state_dict,
    get_gpt_dolomite_tensor_parallel_state_dict,
    unshard_gpt_dolomite_tensor_parallel_state_dicts,
)


_TENSOR_PARALLEL_STATE_DICT_FUNCTIONS = {
    GPTDolomiteConfig.model_type: get_gpt_dolomite_tensor_parallel_state_dict,
}


def get_tensor_parallel_state_dict(
    config: CommonConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    tensor_parallel_word_embeddings: bool,
) -> dict:
    function = _TENSOR_PARALLEL_STATE_DICT_FUNCTIONS[config.model_type]
    return function(
        config,
        safetensors_weights_manager=safetensors_weights_manager,
        tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
    )


_FIX_UNSHARDED_STATE_DICT_FUNCTIONS = {
    GPTDolomiteConfig.model_type: fix_gpt_dolomite_unsharded_state_dict,
}


def fix_unsharded_state_dict(
    config: CommonConfig, state_dict: dict, tensor_parallel_size: int, prefix: str = ""
) -> dict:
    function = _FIX_UNSHARDED_STATE_DICT_FUNCTIONS[config.model_type]
    return function(config, state_dict=state_dict, tensor_parallel_size=tensor_parallel_size, prefix=prefix)


_UNSHARD_TENSOR_PARALLEL_STATE_DICT_FUNCTIONS = {
    GPTDolomiteConfig.model_type: unshard_gpt_dolomite_tensor_parallel_state_dicts,
}


def unshard_tensor_parallel_state_dicts(
    config: GPTDolomiteConfig,
    tensor_parallel_state_dicts: list[dict],
    tensor_parallel_word_embeddings: bool,
    prefix: str = "",
    check_correctness: bool = True,
) -> dict:
    function = _UNSHARD_TENSOR_PARALLEL_STATE_DICT_FUNCTIONS[config.model_type]
    return function(
        config,
        tensor_parallel_state_dicts=tensor_parallel_state_dicts,
        tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
        prefix=prefix,
        check_correctness=check_correctness,
    )
