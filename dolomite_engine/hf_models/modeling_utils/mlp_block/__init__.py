from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...config import CommonConfig
from .mlp import MLP, interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp
from .moe import AuxFreeMoE, MoE, ParameterizedExperts, ParameterizedScatteredExperts, ScatterMoE


def get_mlp_block(config: CommonConfig, use_padding_free_transformer: bool, layer_idx: int) -> MLP | MoE:
    mlp_block_type = config.mlp_blocks[layer_idx]["mlp_block_type"]

    if mlp_block_type == "MLP":
        return MLP(config)
    elif mlp_block_type == "MoE":
        mlp_block_class = ScatterMoE if is_kernel_allowed(Kernel.scattermoe) else MoE
        return mlp_block_class(config, use_padding_free_transformer)
    elif mlp_block_type == "AuxFreeMoE":
        assert is_kernel_allowed(Kernel.scattermoe)
        return AuxFreeMoE(config, use_padding_free_transformer)
    else:
        raise ValueError(f"invalid mlp_block_type ({mlp_block_type}) for layer ({layer_idx})")
