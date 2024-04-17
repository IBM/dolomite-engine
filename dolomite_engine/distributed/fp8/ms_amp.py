from typing import Tuple

import torch.nn as nn
from torch.optim import Optimizer

from ...utils import is_ms_amp_available


if is_ms_amp_available():
    import dolomite_engine.distributed.fp8.ms_amp as ms_amp


def ms_amp_fp8(model: nn.Module, optimizer: Optimizer, optimization_level: int) -> Tuple[nn.Module, Optimizer]:
    model, optimizer = ms_amp.initialize(model, optimizer, opt_level=optimization_level)
    return model, optimizer
