from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...config import CommonConfig
from .auxfree import AuxFreeMoE
from .base import MoE
from .scatter import ScatterMoE


def get_moe(config: CommonConfig, use_aux_free_moe: bool, use_padding_free_transformer: bool, layer_idx: int) -> MoE:
    if use_aux_free_moe:
        assert is_kernel_allowed(Kernel.scattermoe)
        return AuxFreeMoE(config, use_padding_free_transformer, layer_idx=layer_idx)

    moe_class = ScatterMoE if is_kernel_allowed(Kernel.scattermoe) else MoE
    return moe_class(config, use_padding_free_transformer, layer_idx=layer_idx)
