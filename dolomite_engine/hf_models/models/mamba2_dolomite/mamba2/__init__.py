from .....enums import Kernel
from .....kernels import is_kernel_allowed
from ..config import Mamba2DolomiteConfig
from .base import Mamba2Base
from .cuda import Mamba2CUDA


def get_mamba2(config: Mamba2DolomiteConfig, layer_idx: int) -> Mamba2Base | Mamba2CUDA:
    if is_kernel_allowed(Kernel.mamba2_ssm):
        mamba2 = Mamba2CUDA(config=config, layer_idx=layer_idx)
    else:
        mamba2 = Mamba2Base(config=config, layer_idx=layer_idx)

    return mamba2
