import torch
import torch.nn as nn
from transformers import DynamicCache

from ...enums import AttentionHeadType
from ...modeling_utils import get_attention_module, get_normalization_function
from ..moe_dolomite.layer import MoEDolomiteBlock
from ..moe_dolomite.moe import get_moe
from ..rnn_dolomite.attention import DeltaNet
from .config import RNNMoEDolomiteConfig


class RNNMoEDolomiteBlock(MoEDolomiteBlock):
    def __init__(
        self,
        config: RNNMoEDolomiteConfig,
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
        self.mlp = get_moe(
            config,
            use_aux_free_moe=config.use_aux_free_moe,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=causal_mask if self.attention_pattern == "a" else attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

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
