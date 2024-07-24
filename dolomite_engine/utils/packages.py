from .logger import warn_rank_0


try:
    import apex

    _IS_APEX_AVAILABLE = True
except ImportError:
    _IS_APEX_AVAILABLE = False

    warn_rank_0("Apex is not installed")


def is_apex_available() -> bool:
    return _IS_APEX_AVAILABLE


try:
    import deepspeed

    _IS_DEEPSPEED_AVAILABLE = True
except ImportError:
    _IS_DEEPSPEED_AVAILABLE = False

    warn_rank_0("DeepSpeed is not installed")


def is_deepspeed_available() -> bool:
    return _IS_DEEPSPEED_AVAILABLE


try:
    import flash_attn

    _IS_FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    _IS_FLASH_ATTENTION_AVAILABLE = False

    warn_rank_0("Flash Attention is not installed")


def is_flash_attention_available() -> bool:
    return _IS_FLASH_ATTENTION_AVAILABLE


try:
    import aim

    _IS_AIM_AVAILABLE = True
except ImportError:
    _IS_AIM_AVAILABLE = False

    warn_rank_0("aim is not installed")


def is_aim_available() -> bool:
    return _IS_AIM_AVAILABLE


try:
    import wandb

    _IS_WANDB_AVAILABLE = True
except ImportError:
    _IS_WANDB_AVAILABLE = False

    warn_rank_0("wandb is not installed")


def is_wandb_available() -> bool:
    return _IS_WANDB_AVAILABLE


try:
    import colorlog

    _IS_COLORLOG_AVAILABLE = True
except ImportError:
    _IS_COLORLOG_AVAILABLE = False

    warn_rank_0("colorlog is not installed")


def is_colorlog_available() -> bool:
    return _IS_COLORLOG_AVAILABLE


try:
    import transformer_engine

    _IS_TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    _IS_TRANSFORMER_ENGINE_AVAILABLE = False

    warn_rank_0("Nvidia transformer engine is not installed")


def is_transformer_engine_available() -> bool:
    return _IS_TRANSFORMER_ENGINE_AVAILABLE


try:
    import msamp

    _IS_MS_AMP_AVAILABLE = True
except ImportError:
    _IS_MS_AMP_AVAILABLE = False

    warn_rank_0("Microsoft AMP is not installed")


def is_ms_amp_available() -> bool:
    return _IS_MS_AMP_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except ImportError:
    _IS_TRITON_AVAILABLE = False

    warn_rank_0("OpenAI triton is not installed")


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE


try:
    import fla

    _IS_FLA_AVAILABLE = True
except ImportError:
    _IS_FLA_AVAILABLE = False

    warn_rank_0(
        "FlashLinearAttention (FLA) is not installed, install from "
        "https://github.com/sustcsonglin/flash-linear-attention/"
    )


def is_fla_available() -> bool:
    return _IS_FLA_AVAILABLE


try:
    import einops

    _IS_EINOPS_AVAILABLE = True
except ImportError:
    _IS_EINOPS_AVAILABLE = False

    warn_rank_0("einops is not installed")


def is_einops_available() -> bool:
    return _IS_EINOPS_AVAILABLE


try:
    import scattermoe

    _IS_SCATTERMOE_AVAILABLE = True
except ImportError:
    _IS_SCATTERMOE_AVAILABLE = False

    warn_rank_0("scattermoe is not installed, install from https://github.com/shawntan/scattermoe")


def is_scattermoe_available() -> bool:
    return _IS_SCATTERMOE_AVAILABLE
