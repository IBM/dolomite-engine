import logging
from importlib.metadata import distributions

from .logger import log_rank_0, warn_rank_0
from .parallel import run_rank_n


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
    import triton

    _IS_TRITON_AVAILABLE = True
except ImportError:
    _IS_TRITON_AVAILABLE = False

    warn_rank_0("OpenAI triton is not installed")


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE


try:
    import einops

    _IS_EINOPS_AVAILABLE = True
except ImportError:
    _IS_EINOPS_AVAILABLE = False

    warn_rank_0("einops is not installed")


def is_einops_available() -> bool:
    return _IS_EINOPS_AVAILABLE


try:
    import cute_kernels

    _IS_CUTE_KERNELS_AVAILABLE = True
except ImportError:
    _IS_CUTE_KERNELS_AVAILABLE = False

    warn_rank_0("cute-kernels is not installed, install from https://github.com/mayank31398/cute-kernels")


def is_cute_kernels_available() -> bool:
    return _IS_CUTE_KERNELS_AVAILABLE


try:
    import causal_conv1d

    _IS_CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    _IS_CAUSAL_CONV1D_AVAILABLE = False

    warn_rank_0("causal-conv1d is not installed")


def is_causal_conv1d_available() -> bool:
    return _IS_CAUSAL_CONV1D_AVAILABLE


try:
    import torchao

    _IS_TORCHAO_AVAILABLE = True
except ImportError:
    _IS_TORCHAO_AVAILABLE = False

    warn_rank_0("torchao is not installed")


def is_torchao_available() -> bool:
    return _IS_TORCHAO_AVAILABLE


@run_rank_n
def log_environment() -> None:
    packages = sorted(["{}=={}".format(d.metadata["Name"], d.version) for d in distributions()])

    log_rank_0(logging.INFO, "------------------------ packages ------------------------")
    for package in packages:
        log_rank_0(logging.INFO, package)
    log_rank_0(logging.INFO, "-------------------- end of packages ---------------------")
