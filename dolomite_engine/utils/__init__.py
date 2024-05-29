import torch
import torch.distributed

from .logging import log_rank_0, print_rank_0, print_ranks_all, set_logger
from .mixed_precision import normalize_dtype_string, string_to_torch_dtype, torch_dtype_to_string
from .packages import (
    is_apex_available,
    is_deepspeed_available,
    is_flash_attention_available,
    is_ms_amp_available,
    is_transformer_engine_available,
    is_triton_available,
)
from .parallel import ProcessGroupManager
from .pydantic import BaseArgs
from .ranks import get_global_rank, get_local_rank, get_world_size, run_rank_n
from .safetensors import SafeTensorsWeightsManager
from .tracking import ExperimentsTracker, ProgressBar, RunningMean
from .wrapper import get_module_class_from_name
from .yaml import load_yaml


def init_distributed() -> None:
    """intialize distributed"""

    torch.distributed.init_process_group("nccl", rank=get_global_rank(), world_size=get_world_size())
    torch.cuda.set_device(get_local_rank())


def setup_tf32(use_tf32: bool = True) -> None:
    """whether to use tf32 instead of fp32

    Args:
        use_tf32 (bool, optional): Defaults to True.
    """

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
