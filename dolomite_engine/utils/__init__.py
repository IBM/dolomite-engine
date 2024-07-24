import logging

import torch
import torch.distributed

from .hf_hub import download_repo
from .logger import log_rank_0, print_rank_0, print_ranks_all, set_logger
from .mixed_precision import normalize_dtype_string, string_to_torch_dtype, torch_dtype_to_string
from .packages import (
    is_apex_available,
    is_deepspeed_available,
    is_einops_available,
    is_fla_available,
    is_flash_attention_available,
    is_ms_amp_available,
    is_scattermoe_available,
    is_transformer_engine_available,
    is_triton_available,
)
from .parallel import ProcessGroupManager, run_rank_n
from .pydantic import BaseArgs
from .random import CUDA_RNGStatesTracker, get_cuda_rng_tracker, set_cuda_rng_tracker
from .safetensors import SafeTensorsWeightsManager
from .tracking import ExperimentsTracker, ProgressBar, RunningMean
from .wrapper import get_module_class_from_name
from .yaml import load_yaml


def init_distributed(
    tensor_parallel_size: int,
    data_parallel_size: int,
    data_parallel_replication_world_size: int,
    data_parallel_sharding_world_size: int,
    timeout_minutes: int = None,
) -> None:
    """intialize distributed

    Args:
        tensor_parallel_size (int): tensor parallel size
        data_parallel_size (int): data parallel size
        data_parallel_replication_world_size (int): data parallel replication world size
        data_parallel_sharding_world_size (int): data parallel sharding world size
        timeout_minutes (int, optional): distributed timeout in minutes. Defaults to None.
    """

    process_group_manager = ProcessGroupManager(
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        data_parallel_replication_world_size=data_parallel_replication_world_size,
        data_parallel_sharding_world_size=data_parallel_sharding_world_size,
        timeout_minutes=timeout_minutes,
    )

    log_rank_0(logging.INFO, process_group_manager)
    log_rank_0(logging.INFO, f"total GPUs = {process_group_manager.get_world_size()}")
    log_rank_0(logging.INFO, f"tensor parallel size = {process_group_manager.get_tensor_parallel_world_size()}")
    print_ranks_all(f"tensor parallel mesh = {process_group_manager.get_tensor_parallel_mesh()}")
    log_rank_0(logging.INFO, f"data parallel size = {process_group_manager.get_data_parallel_world_size()}")
    print_ranks_all(f"data parallel mesh = {process_group_manager.get_data_parallel_mesh()}")


def setup_tf32(use_tf32: bool = True) -> None:
    """whether to use tf32 instead of fp32

    Args:
        use_tf32 (bool, optional): Defaults to True.
    """

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
