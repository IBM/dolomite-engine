from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...config import CommonConfig
from .mlp import MLP_TP
from .moe import ScatterMoE_TP


def get_mlp_block_TP(
    config: CommonConfig, use_padding_free_transformer: bool, sequence_parallel: bool, layer_idx: int
) -> MLP_TP | ScatterMoE_TP:
    mlp_block_type = config.mlp_blocks[layer_idx]["mlp_block_type"]

    if mlp_block_type == "MLP":
        return MLP_TP(
            config, use_padding_free_transformer=use_padding_free_transformer, sequence_parallel=sequence_parallel
        )
    elif mlp_block_type == "MoE":
        assert is_kernel_allowed(Kernel.scattermoe)
        return ScatterMoE_TP(
            config, use_padding_free_transformer=use_padding_free_transformer, sequence_parallel=sequence_parallel
        )
    else:
        raise ValueError(f"invalid mlp_block_type ({mlp_block_type}) for layer ({layer_idx})")
