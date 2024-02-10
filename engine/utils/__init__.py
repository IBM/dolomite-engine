import torch
import torch.distributed as dist

from .logging import AimTracker, ExperimentsTracker, ProgressBar, RunningMean, print_rank_0, warn_rank_0
from .monitoring import register_profiler, register_timer
from .ranks import get_global_rank, get_local_rank, get_world_size, run_rank_n
from .yaml import load_yaml


def init_distributed() -> None:
    """intialize distributed"""

    dist.init_process_group("nccl", rank=get_global_rank(), world_size=get_world_size())
    torch.cuda.set_device(get_local_rank())
    print_rank_0(f"total GPUs = {get_world_size()}")


def setup_tf32(use_tf32: bool = True) -> None:
    """whether to use tf32 instead of fp32

    Args:
        use_tf32 (bool, optional): Defaults to True.
    """

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
