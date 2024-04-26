import torch
import torch.nn as nn


def is_apex_layernorm_available() -> bool:
    try:
        from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

        return True
    except ImportError:
        return False


if is_apex_layernorm_available():
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction


class ApexLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: int, eps: float = 0.00001) -> None:
        if not is_apex_layernorm_available():
            raise ImportError("build apex from source")

        super().__init__(normalized_shape, eps=eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps, True)
