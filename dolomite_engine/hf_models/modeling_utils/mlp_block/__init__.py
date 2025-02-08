from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...config import CommonConfig
from .mlp import MLP, interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp
from .moe import AuxFreeMoE, MoE, ParameterizedExperts, ParameterizedScatteredExperts, ScatterMoE


def get_mlp_block(config: CommonConfig, use_padding_free_transformer: bool, layer_idx: int) -> MLP | MoE:
    mlp_block_type = config.mlp_blocks[layer_idx]["mlp_block_type"]
    mlp_block_args = config.mlp_blocks_args[layer_idx]

    kwargs = dict(
        hidden_size=config.hidden_size,
        intermediate_size=mlp_block_args.intermediate_size,
        activation_function=mlp_block_args.activation_function,
        add_bias=mlp_block_args.add_bias,
        residual_dropout=mlp_block_args.resid_pdrop,
        init_method=mlp_block_args.init_method,
        initializer_range=mlp_block_args.initializer_range,
        m_width=mlp_block_args.m_width,
        num_layers=config.num_layers,
    )

    if mlp_block_type == "MLP":
        return MLP(**kwargs)
    elif mlp_block_type == "MoE":
        mlp_block_class = ScatterMoE if is_kernel_allowed(Kernel.scattermoe) else MoE
        return mlp_block_class(
            **kwargs,
            shared_intermediate_size=mlp_block_args.shared_intermediate_size,
            num_experts=mlp_block_args.num_experts,
            num_experts_per_tok=mlp_block_args.num_experts_per_tok,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif mlp_block_type == "AuxFreeMoE":
        assert is_kernel_allowed(Kernel.scattermoe)
        return AuxFreeMoE(config, use_padding_free_transformer)
    else:
        raise ValueError(f"invalid mlp_block_type ({mlp_block_type}) for layer ({layer_idx})")
