from ..config import MoEDolomiteConfig
from .auxfree import AuxFreeMoE
from .base import MoE
from .scatter import ScatterMoE


_MOE_MODULES = {"eager": MoE, "scattermoe": ScatterMoE, "auxfreemoe": AuxFreeMoE}


def get_moe(
    config: MoEDolomiteConfig,
    moe_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> MoE:
    if moe_implementation in _MOE_MODULES:
        return _MOE_MODULES[moe_implementation](config, use_padding_free_transformer, layer_idx=layer_idx)

    raise ValueError(f"unexpected `moe_implementation` {moe_implementation}")
