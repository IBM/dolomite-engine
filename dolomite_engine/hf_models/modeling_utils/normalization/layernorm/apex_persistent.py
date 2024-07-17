import torch
import torch.nn as nn


def is_apex_persistent_layernorm_available() -> bool:
    try:
        from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

        return True
    except ImportError:
        return False


if is_apex_persistent_layernorm_available():
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN


_PERSISTENT_LAYERNORM_ALLOWED_HIDDEN_STATES = [
    1024,
    1536,
    2048,
    2304,
    3072,
    3840,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    12800,
    15360,
    16384,
    18432,
    20480,
    24576,
    25600,
    30720,
    32768,
    40960,
    49152,
    65536,
]


def apex_persistent_layernorm(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, memory_efficient
) -> torch.Tensor:
    return FastLayerNormFN.apply(input, weight, bias, eps, memory_efficient)


class ApexPersistentLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: int, eps: float = 0.00001) -> None:
        if not is_apex_persistent_layernorm_available():
            raise ImportError("build apex from source with --fast_layer_norm")

        super().__init__(normalized_shape, eps=eps)

        assert (
            self.normalized_shape[0] in _PERSISTENT_LAYERNORM_ALLOWED_HIDDEN_STATES
        ), "persistent layernorm kernel is not avilable for the specified hidden dimension"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apex_persistent_layernorm(input, self.weight, self.bias, self.eps, True)
