import torch.nn as nn

from ...modeling_utils import get_attention_module, get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from .config import MoEDolomiteConfig
from .moe import get_moe


class MoEDolomiteBlock(GPTDolomiteBlock):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp = get_moe(
            config,
            use_aux_free_moe=config.use_aux_free_moe,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
        )
