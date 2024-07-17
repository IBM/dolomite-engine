import numbers

import torch
import torch.nn as nn

from .....modeling_utils import RMSNorm


class EnsembleRMSNorm(RMSNorm):
    def __init__(self, normalized_shape: int, tp_world_size: int, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)

        self.weight = nn.Parameter(torch.ones(tp_world_size, 1, 1, normalized_shape))
        self.eps = eps

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
