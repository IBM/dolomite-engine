from ...modeling_utils import get_attention_module, get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from ..gpt_dolomite.mlp import MLP
from .config import Mamba2DolomiteConfig
from .mamba2 import get_mamba2


class Mamba2DolomiteBlock(GPTDolomiteBlock):
    def __init__(
        self,
        config: Mamba2DolomiteConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        self.is_attention_layer = config.layer_map[layer_idx] == "attention"
        self.is_mamba_layer = config.layer_map[layer_idx] == "mamba2"

        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )

        if self.is_attention_layer:
            self.attn = get_attention_module(
                config, True, attention_implementation, use_padding_free_transformer, layer_idx
            )
        elif self.is_mamba_layer:
            self.attn = get_mamba2(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"unexpected layer_map value for layer {layer_idx}")

        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp = MLP(config)
