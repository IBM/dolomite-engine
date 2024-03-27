import torch.nn as nn

from ...enums import PositionEmbeddingType
from ...modeling_utils import get_normalization_function
from ..moe_megablocks.layer import SparseMoEBlock
from .config import DenseMoEConfig
from .moa import DenseMoA_SDPA
from .moe import DenseMoE


class DenseMoEBlock(SparseMoEBlock):
    def __init__(
        self,
        config: DenseMoEConfig,
        normalization_implementation: str,
        layer_idx: int = None,
        inference_method: dict = None,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.layer_idx = layer_idx

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.attn = DenseMoA_SDPA(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_experts=config.num_experts,
            position_embedding_type=PositionEmbeddingType(config.position_embedding_type),
            causal=True,
            add_bias=config.add_bias,
            scale_attention_weights=config.scale_attn_weights,
            attention_softmax_in_fp32=config.attention_softmax_in_fp32,
            scale_attention_softmax_in_fp32=config.scale_attention_softmax_in_fp32,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            layer_idx=layer_idx,
            inference_method=inference_method,
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.mlp = DenseMoE(
            config.hidden_size,
            intermediate_size=self.inner_dim,
            num_experts=config.num_experts,
            activation_function=config.activation_function,
            add_bias=config.add_bias,
            residual_dropout=config.resid_pdrop,
            inference_method=inference_method,
        )
