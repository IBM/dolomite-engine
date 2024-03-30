import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..enums import LRDecaySchedule


def get_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_constant_steps: int,
    num_decay_steps: int,
    num_training_steps: int,
    lr_decay_style: LRDecaySchedule,
    lr_decay_factor: float,
) -> LambdaLR:
    lr_warmup_boundary = num_warmup_steps
    lr_constant_boundary = lr_warmup_boundary + num_constant_steps

    lr_decay_boundary = num_training_steps
    if num_decay_steps is not None:
        lr_decay_boundary = lr_constant_boundary + num_decay_steps

    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=partial(
            get_factor,
            lr_warmup_boundary=lr_warmup_boundary,
            lr_constant_boundary=lr_constant_boundary,
            lr_decay_boundary=lr_decay_boundary,
            lr_decay_style=lr_decay_style,
            lr_decay_factor=lr_decay_factor,
        ),
    )
    return lr_scheduler


def get_factor(
    num_steps: int,
    lr_warmup_boundary: int,
    lr_constant_boundary: int,
    lr_decay_boundary: int,
    lr_decay_style: LRDecaySchedule,
    lr_decay_factor: float,
) -> float:
    # Use linear warmup for the initial part.
    if lr_warmup_boundary > 0 and num_steps <= lr_warmup_boundary:
        return _linear(m=1 / lr_warmup_boundary, c=0, x=num_steps)

    if num_steps > lr_warmup_boundary and num_steps <= lr_constant_boundary:
        return 1

    if lr_decay_style == LRDecaySchedule.constant:
        return 1
    elif lr_decay_style == LRDecaySchedule.exponential:
        return _exponential(
            a=(1 - lr_decay_factor) * math.e / (math.e - 1),
            b=(lr_decay_factor * math.e - 1) / (math.e - 1),
            t=lr_decay_boundary - lr_constant_boundary,
            x=num_steps - lr_constant_boundary,
        )

    if num_steps > lr_constant_boundary and num_steps <= lr_decay_boundary:
        if lr_decay_style == LRDecaySchedule.linear:
            return _linear(
                m=(lr_decay_factor - 1) / (lr_decay_boundary - lr_constant_boundary),
                c=1,
                x=num_steps - lr_constant_boundary,
            )
        elif lr_decay_style == LRDecaySchedule.cosine:
            return _cosine(
                a=1 - lr_decay_factor,
                b=lr_decay_factor,
                t=lr_decay_boundary - lr_constant_boundary,
                x=num_steps - lr_constant_boundary,
            )
        else:
            raise Exception("{} decay style is not supported.".format(self.lr_decay_style))

    return lr_decay_factor


def _linear(m: float, c: float, x: float) -> float:
    return m * x + c


def _exponential(a: float, b: float, t: float, x: float) -> float:
    return a * math.exp(-x / t) + b


def _cosine(a: float, b: float, t: float, x: float) -> float:
    return a * (1 + math.cos(math.pi * x / t)) / 2 + b
