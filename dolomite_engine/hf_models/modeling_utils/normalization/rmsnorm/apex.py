import torch
import torch.nn as nn


def is_apex_rmsnorm_available() -> bool:
    try:
        from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction

        return True
    except ImportError:
        return False


if is_apex_rmsnorm_available():
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction


def apex_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float, memory_efficient: bool) -> torch.Tensor:
    normalized_shape = (input.shape[-1],)
    return FusedRMSNormAffineMixedDtypesFunction.apply(input, weight, normalized_shape, eps, memory_efficient)


class ApexRMSNorm(nn.RMSNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        if not is_apex_rmsnorm_available():
            raise ImportError("build apex from source")

        super().__init__(normalized_shape, eps=eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apex_rmsnorm(input, self.weight, self.eps, True)
