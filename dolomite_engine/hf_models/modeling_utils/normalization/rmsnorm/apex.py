import torch

from .base import RMSNorm


def is_apex_rmsnorm_available() -> bool:
    try:
        from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction

        return True
    except ImportError:
        return False


if is_apex_rmsnorm_available():
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction


class ApexRMSNorm(RMSNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        if not is_apex_rmsnorm_available():
            raise ImportError("build apex from source")

        super().__init__(normalized_shape, eps=eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = FusedRMSNormAffineMixedDtypesFunction.apply(input, self.weight, self.normalized_shape, self.eps, True)
        return input
