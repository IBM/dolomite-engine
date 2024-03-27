from typing import Iterable, Tuple

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .optimizer import get_optimizer
from .scheduler import get_scheduler


def get_optimizer_and_lr_scheduler(
    optimizer_class_name: str,
    optimizer_class_args: dict,
    cpu_offload: bool,
    trainable_parameters: Iterable[nn.Parameter],
    lr_schedule: str,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Tuple[Optimizer, LambdaLR]:
    optimizer = get_optimizer(
        optimizer_class_name=optimizer_class_name,
        optimizer_class_args=optimizer_class_args,
        cpu_offload=cpu_offload,
        parameters=trainable_parameters,
    )

    lr_scheduler = get_scheduler(
        schedule=lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler
