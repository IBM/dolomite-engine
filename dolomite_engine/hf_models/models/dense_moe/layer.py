import torch.nn as nn

from ...modeling_utils import get_normalization_function
from ..moe_dolomite.layer import SparseMoEBlock
from .config import DenseMoEConfig
from .moa import DenseMoA_SDPA
from .moe import DenseMoE


class DenseMoEBlock(SparseMoEBlock):
    def __init__(
        self,
        config: DenseMoEConfig,
        normalization_implementation: str,
        layer_idx: int | None = None,
        inference_method: dict | None = None,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.attn = DenseMoA_SDPA(config, causal=True, layer_idx=layer_idx, inference_method=inference_method)
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.moe = DenseMoE(config, inference_method=inference_method)

        self.mlp = None
