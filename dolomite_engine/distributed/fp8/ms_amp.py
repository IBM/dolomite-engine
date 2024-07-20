import torch.nn as nn
from torch.optim import Optimizer

from ...utils import is_ms_amp_available


if is_ms_amp_available():
    import msamp


def ms_amp_fp8(model: nn.Module, optimizer: Optimizer, optimization_level: int) -> tuple[nn.Module, Optimizer]:
    model, optimizer = msamp.initialize(model, optimizer, opt_level=optimization_level)
    return model, optimizer
