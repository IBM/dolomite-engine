from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


_SCHEDULER_CLASSES = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
}


def get_scheduler(schedule: str, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """setup learning rate scheduler

    Args:
        schedule (str): schedule type
        optimizer (Optimizer): an optimizer
        num_warmup_steps (int): number of steps for warmup
        num_training_steps (int): number of steps for training

    Returns:
        Callable: learning rate scheduler
    """

    if schedule not in _SCHEDULER_CLASSES:
        raise ValueError(f"invalid schedule ({schedule})")

    lr_scheduler_function = _SCHEDULER_CLASSES[schedule]

    lr_scheduler = lr_scheduler_function(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    return lr_scheduler
