from argparse import Namespace
from typing import List, Tuple

import deepspeed
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.utils.distributed import get_local_rank, get_world_size
from src.utils.logging import print_rank_0
from src.utils.monitoring import register_timer


def init_distributed(dist_backend: str = "nccl") -> None:
    """intialize distributed

    Args:
        dist_backend (str, optional): backend to use. Defaults to "nccl".
    """

    deepspeed.init_distributed(dist_backend=dist_backend)
    torch.cuda.set_device(get_local_rank())
    print_rank_0(f"total GPUs = {get_world_size()}")


@register_timer("deepspeed_initialize")
def deepspeed_initialize(
    args: Namespace,
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    datasets: List[Dataset],
) -> Tuple[deepspeed.DeepSpeedEngine, List[DataLoader]]:
    """converts the model to a ZeRO-DP sharded model

    Args:
        args (Namespace): arguments based on training / inference mode
        model (torch.nn.Module): any torch.nn.Module object
        optimizer (Optimizer): an optimizer, preferably one that is supported by deepspeed
        lr_scheduler (LRScheduler): any learning rate scheduler
        datasets (List[Dataset]): a list of datasets in which the first element is the training dataset
        train_sampler (DistributedSampler): data sampler to use for train split

    Returns:
        Tuple[deepspeed.DeepSpeedEngine, List[DataLoader]]: sharded model and datasets
    """

    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=get_deepspeed_config(args),
    )
    model: deepspeed.DeepSpeedEngine

    dataloaders = []

    from src.data.dataset import ConcatenatedDatasets
    from src.data.sampler import ConcatenatedDataSampler

    train_dataset = datasets[0]
    train_dataloader = model.deepspeed_io(
        train_dataset, data_sampler=ConcatenatedDataSampler(args, train_dataset), collate_fn=train_dataset.collate_fn
    )
    dataloaders.append(train_dataloader)

    for dataset in datasets[1:]:
        dataloader = None
        if dataset is not None:
            dataloader = model.deepspeed_io(
                dataset,
                data_sampler=ConcatenatedDataSampler(args, dataset, shuffle=False),
                collate_fn=dataset.collate_fn,
            )

        dataloaders.append(dataloader)

    return model, dataloaders


def get_deepspeed_config(args: Namespace) -> dict:
    """generate deepspeed config from the args

    Args:
        args (Namespace): arguments based on training / inference mode

    Returns:
        dict: deepspeed config
    """

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

    # cpu offload
    if args.cpu_offload:
        config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    from src.utils.monitoring import is_debugging_enabled

    # debugging stuff
    config["steps_per_print"] = np.inf
    if is_debugging_enabled():
        config["steps_per_print"] = args.steps_per_print

    return config


def setup_tf32(use_tf32: bool = True) -> None:
    """whether to use tf32 instead of fp32

    Args:
        use_tf32 (bool, optional): Defaults to True.
    """

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
