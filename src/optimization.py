from typing import Callable, Tuple

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from src.constants import LearningRateScheduler
from src.utils import register_profiler, register_timer


@register_profiler("setup_optimizer")
@register_timer("setup_optimizer")
def get_optimizer(
    cpu_offload: bool, parameters: list, lr: float, weight_decay: float, betas: Tuple[float, float], eps: float
) -> Callable:
    """setup optimizer for the model

    Args:
        cpu_offload (bool): whether to use CPU offloading
        parameters (list): list of model parameters
        lr (float): starting learning rate
        weight_decay (float): weight decay parameter
        betas (Tuple[float, float]): _description_
        eps (float): epsilon

    Returns:
        Callable: Adam optimizer
    """

    if cpu_offload:
        # DeepSpeedCPUAdam is faster with CPU offloading
        from deepspeed.ops.adam import DeepSpeedCPUAdam as Adam
    else:
        from apex.optimizers import FusedAdam as Adam

    optimizer = Adam(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    return optimizer


def get_scheduler_method(schedule: LearningRateScheduler) -> Callable:
    """setup learning rate scheduler

    Args:
        schedule (LearningRateScheduler): schedule type

    Returns:
        Callable: learning rate scheduler
    """

    if schedule == LearningRateScheduler.linear:
        return get_linear_schedule_with_warmup
    elif schedule == LearningRateScheduler.cosine:
        return get_cosine_schedule_with_warmup
