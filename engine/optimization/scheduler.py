from typing import Callable

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from engine.constants import LearningRateScheduler


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
