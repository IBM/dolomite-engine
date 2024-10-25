import logging

import torch
import torch.distributed

from .hf_hub import download_repo
from .logger import log_metrics, log_rank_0, print_rank_0, print_ranks_all, set_logger
from .loss_dict import MetricsTrackingDict
from .miscellaneous import divide_if_divisible
from .mixed_precision import normalize_dtype_string, string_to_torch_dtype, torch_dtype_to_string
from .packages import (
    is_apex_available,
    is_causal_conv1d_available,
    is_einops_available,
    is_fla_available,
    is_flash_attention_available,
    is_kernel_hyperdrive_available,
    is_ms_amp_available,
    is_transformer_engine_available,
    is_triton_available,
    log_environment,
)
from .parallel import ProcessGroupManager, get_pipeline_stage_ids_on_current_rank, run_rank_n
from .pydantic import BaseArgs
from .safetensors import SafeTensorsWeightsManager
from .tracking import ExperimentsTracker, ProgressBar
from .wrapper import get_module_class_from_name
from .yaml import load_yaml


def init_distributed(
    tensor_parallel_world_size: int,
    pipeline_parallel_world_size: int,
    data_parallel_size: int,
    data_parallel_replication_world_size: int,
    data_parallel_sharding_world_size: int,
    zero_stage: int,
    timeout_minutes: int = None,
    use_async_tensor_parallel: bool = False,
) -> None:
    """intialize distributed

    Args:
        tensor_parallel_world_size (int): tensor parallel size
        pipeline_parallel_world_size (int): pipeline parallel size
        data_parallel_size (int): data parallel size
        data_parallel_replication_world_size (int): data parallel replication world size
        data_parallel_sharding_world_size (int): data parallel sharding world size
        zero_stage (int): zero stage
        timeout_minutes (int, optional): distributed timeout in minutes. Defaults to None.
        use_async_tensor_parallel (bool): whether to use async-TP. Defaults to False.
    """

    process_group_manager = ProcessGroupManager(
        tensor_parallel_world_size=tensor_parallel_world_size,
        pipeline_parallel_world_size=pipeline_parallel_world_size,
        data_parallel_size=data_parallel_size,
        data_parallel_replication_world_size=data_parallel_replication_world_size,
        data_parallel_sharding_world_size=data_parallel_sharding_world_size,
        zero_stage=zero_stage,
        timeout_minutes=timeout_minutes,
        use_async_tensor_parallel=use_async_tensor_parallel,
    )

    log_rank_0(logging.INFO, process_group_manager)
    log_rank_0(logging.INFO, f"total GPUs = {process_group_manager.get_world_size()}")
    log_rank_0(logging.INFO, f"tensor parallel size = {process_group_manager.get_tensor_parallel_world_size()}")
    log_rank_0(logging.INFO, f"data parallel size = {process_group_manager.get_data_parallel_world_size()}")


def setup_tf32(use_tf32: bool = True) -> None:
    """whether to use tf32 instead of fp32

    Args:
        use_tf32 (bool, optional): Defaults to True.
    """

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
