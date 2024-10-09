import torch.nn as nn

from ...enums import AttentionHeadType
from ...modeling_utils import get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from ..gpt_dolomite.mlp import MLP
from .attention import DeltaNet, RNNFlashAttention2
from .config import RNNDolomiteConfig


class RNNDolomiteBlock(GPTDolomiteBlock):
    """
    Layer implementation for the transformer block
    """

    def __init__(
        self,
        config: RNNDolomiteConfig,
        normalization_implementation: str,
        attention_pattern: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        nn.Module.__init__(self)

        assert not use_padding_free_transformer

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )

        if attention_pattern == "flash_attention_2":
            self.attn = RNNFlashAttention2(config, True, layer_idx)
        elif attention_pattern == "deltanet":
            self.attn = DeltaNet(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Attention pattern {attention_pattern} not supported.")

        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )

        self.mlp = MLP(config)
