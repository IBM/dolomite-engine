from argparse import Namespace
from typing import List, Tuple

import deepspeed
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.distributed import get_local_rank, get_world_size
from src.utils.logging import print_rank_0
from src.utils.monitoring import register_timer


@register_timer("init_distributed")
def init_distributed(dist_backend: str = "nccl") -> None:
    deepspeed.init_distributed(dist_backend=dist_backend)
    torch.cuda.set_device(get_local_rank())
    print_rank_0(f"total GPUs = {get_world_size()}")


@register_timer("deepspeed_initialize")
def deepspeed_initialize(
    args: Namespace, model: torch.nn.Module, optimizer, lr_scheduler, datasets: List[Dataset]
) -> Tuple[deepspeed.DeepSpeedEngine, List[DataLoader]]:
    """
    Converts the model to a ZeRO-DP sharded model

    Args:
        model (Model): any torch.nn.Module object
        optimizer (Optimizer): an optimizer, preferably one that is supported by deepspeed
        lr_scheduler (LRScheduler): any learning rate scheduler
        datasets (List[Dataset]): a list of datasets in which the first element is the training dataset

    Returns:
        Tuple[torch.nn.Module, List[DataLoader]]: sharded model and datasets
    """
    # first dataset should be training dataset and is sharded using a distributed random sampler
    train_dataset = datasets[0]
    model, _, train_dataset, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        training_data=train_dataset,
        lr_scheduler=lr_scheduler,
        config=get_deepspeed_config(args),
    )
    datasets[0] = train_dataset

    # other datasets only need a sequential sampler
    for i in range(1, len(datasets)):
        if datasets[i] is not None:
            datasets[i] = model.deepspeed_io(datasets[i], route="eval")

    return model, datasets


def get_deepspeed_config(args: Namespace) -> dict:
    config = {
        "zero_optimization": {
            "stage": args.stage,
            "overlap_comm": args.overlap_comm,
            "contiguous_gradients": args.contiguous_gradients,
        },
        "train_micro_batch_size_per_gpu": args.batch_size_per_gpu,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    # mixed precision options
    if args.dtype == torch.bfloat16:
        config["bf16"] = {"enabled": True}
    elif args.dtype == torch.float16:
        config["fp16"] = {"enabled": True, "auto_cast": True}

    if args.cpu_offload:
        config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    from src.utils.monitoring import is_debugging_enabled

    config["steps_per_print"] = np.inf
    if is_debugging_enabled():
        config["steps_per_print"] = args.steps_per_print

    return config


def setup_tf32(use_tf32: bool = True) -> None:
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
