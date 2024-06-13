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


def apex_layernorm(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, memory_efficient: bool
) -> torch.Tensor:
    normalized_shape = (input.shape[-1],)
    return FusedLayerNormAffineFunction.apply(input, weight, bias, normalized_shape, eps, memory_efficient)


class ApexLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: int, eps: float = 0.00001) -> None:
        if not is_apex_layernorm_available():
            raise ImportError("build apex from source")

        super().__init__(normalized_shape, eps=eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apex_layernorm(input, self.weight, self.bias, self.eps, True)
