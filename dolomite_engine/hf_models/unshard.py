from .config import CommonConfig
from .models import (
    GPTDolomiteConfig,
    MoEDolomiteConfig,
    fix_gpt_dolomite_unsharded_state_dict,
    fix_moe_dolomite_unsharded_state_dict,
    unshard_gpt_dolomite_tensor_parallel_state_dicts,
    unshard_moe_dolomite_tensor_parallel_state_dicts,
)


_UNSHARD_STATE_DICT_FUNCTIONS = {
    GPTDolomiteConfig.model_type: unshard_gpt_dolomite_tensor_parallel_state_dicts,
    MoEDolomiteConfig.model_type: unshard_moe_dolomite_tensor_parallel_state_dicts,
}


def unshard_tensor_parallel_state_dicts(
    config: MoEDolomiteConfig,
    tensor_parallel_state_dicts: list[dict],
    tensor_parallel_word_embeddings: bool,
    prefix: str = "",
    check_correctness: bool = True,
) -> dict:
    if config.model_type in _UNSHARD_STATE_DICT_FUNCTIONS:
        return _UNSHARD_STATE_DICT_FUNCTIONS[config.model_type](
            config=config,
            tensor_parallel_state_dicts=tensor_parallel_state_dicts,
            tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
            prefix=prefix,
            check_correctness=check_correctness,
        )

    raise ValueError(f"unsupported `model_type` ({config.model_type})")


_FIX_UNSHARDED_STATE_DICT_FUNCTIONS = {
    GPTDolomiteConfig.model_type: fix_gpt_dolomite_unsharded_state_dict,
    MoEDolomiteConfig.model_type: fix_moe_dolomite_unsharded_state_dict,
}


def fix_unsharded_state_dict(
    config: CommonConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str = ""
) -> dict:
    if config.model_type in _FIX_UNSHARDED_STATE_DICT_FUNCTIONS:
        return _FIX_UNSHARDED_STATE_DICT_FUNCTIONS[config.model_type](
            config=config, state_dict=state_dict, tensor_parallel_world_size=tensor_parallel_world_size, prefix=prefix
        )

    raise ValueError(f"unsupported `model_type` ({config.model_type})")
