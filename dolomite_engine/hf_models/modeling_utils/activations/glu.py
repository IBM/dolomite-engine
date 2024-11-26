import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_cute_kernels_available
from .base import get_base_activation


if is_cute_kernels_available():
    from cute_kernels.kernels import swiglu_cute


_GLU_BASE_MAPPING = {
    "ceglu": "celu",
    "eglu": "elu",
    "geglu": "gelu",
    "miglu": "mish",
    "mishglu": "mish",
    "preglu": "prelu",
    "reglu": "relu",
    "rreglu": "rrelu",
    "seglu": "selu",
    "swiglu": "swish",
}


class GLUActivation(nn.Module):
    def __init__(self, base_activation: nn.Module) -> None:
        super().__init__()
        self.base_activation = base_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.chunk(2, dim=-1)
        return x[0] * self.base_activation(x[1])


class CuteSwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.chunk(2, dim=-1)
        return swiglu_cute(gate=x[1], up=x[0])


def get_glu_activation(name: str) -> nn.Module:
    # for glu and sigmoid_glu, we directly return the pytorch's GLU
    if name in ["glu", "sigmoid_glu"]:
        activation_function = nn.GLU()
    elif is_kernel_allowed(Kernel.cute_swiglu) and name in ["swiglu", "swish_glu"]:
        activation_function = CuteSwiGLU()
    else:
        if name in _GLU_BASE_MAPPING:
            name = _GLU_BASE_MAPPING[name]
        elif name.endswith("_glu"):
            name = name.rstrip("_glu")
        else:
            raise ValueError("invalid activation function")

        base_activation = get_base_activation(name)
        activation_function = GLUActivation(base_activation)

    return activation_function


def is_glu(name: str) -> bool:
    return name.endswith("glu")
