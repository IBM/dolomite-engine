import torch.nn as nn

from ...enums import AttentionHeadType
from ...modeling_utils import get_attention_module, get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from ..gpt_dolomite.mlp import MLP
from .attention import DeltaNet
from .config import RNNDolomiteConfig


class RNNDolomiteBlock(GPTDolomiteBlock):
    def __init__(
        self,
        config: RNNDolomiteConfig,
        attention_implementation: str,
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
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )

        self.attention_pattern = attention_pattern

        if attention_pattern == "a":
            self.attn = get_attention_module(
                config=config,
                causal=True,
                attention_implementation=attention_implementation,
                use_padding_free_transformer=use_padding_free_transformer,
                layer_idx=layer_idx,
            )
        elif attention_pattern == "d":
            self.attn = DeltaNet(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Attention pattern {attention_pattern} not supported.")

        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )

        self.mlp = MLP(config)
