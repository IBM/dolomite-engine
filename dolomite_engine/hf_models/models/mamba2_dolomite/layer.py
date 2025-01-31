import torch
import torch.nn as nn
from transformers import DynamicCache

from ...modeling_utils import get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
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
        self.is_mamba_layer = config.layer_map[layer_idx] == "mamba"

        if self.is_attention_layer:
            super().__init__(
                config=config,
                attention_implementation=attention_implementation,
                use_padding_free_transformer=use_padding_free_transformer,
                layer_idx=layer_idx,
            )
        elif self.is_mamba_layer:
            nn.Module.__init__()

            self.ln = get_normalization_function(
                config.normalization_function, config.hidden_size, eps=config.layer_norm_epsilon
            )
            self.mamba = get_mamba2(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"unexpected layer_map value for layer {layer_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        cache_position: int | None = None,
    ) -> tuple[torch.Tensor]:
        if self.is_attention_layer:
            hidden_states = super().forward(
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.is_mamba_layer:
            residual = hidden_states

            hidden_states = self.ln(hidden_states)
            hidden_states = self.mamba(
                hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

            hidden_states = residual + hidden_states
        else:
            raise ValueError(f"unexpected layer_map value for layer {layer_idx}")

        return hidden_states
