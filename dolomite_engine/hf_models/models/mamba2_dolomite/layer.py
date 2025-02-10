import torch
import torch.nn as nn

from ...modeling_utils import get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from .cache import HybridMambaAttentionDynamicCache
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
        nn.Module.__init__(self)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: HybridMambaAttentionDynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        if self.is_attention_layer:
            hidden_states = self.attn(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.is_mamba_layer:
            hidden_states = self.attn(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)
        else:
            raise ValueError(f"unexpected layer_map value for layer {self.layer_idx}")

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        # residual connection
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        # residual connection
        hidden_states = hidden_states + residual

        return hidden_states
