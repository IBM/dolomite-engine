from copy import deepcopy

import torch
import torch.nn as nn
from transformers import DynamicCache

from ...enums import AttentionHeadType
from ...modeling_utils import get_attention_module, get_normalization_function
from ..gpt_dolomite.mlp import MLP
from ..moe_dolomite.layer import SparseMoEBlock
from ..moe_dolomite.moe import get_moe
from ..rnn_dolomite.attention import DeltaNet
from .config import RNNMoEDolomiteConfig


class RNNMoEDolomiteBlock(SparseMoEBlock):
    def __init__(
        self,
        config: RNNMoEDolomiteConfig,
        attention_implementation: str,
        attention_pattern: str,
        use_padding_free_transformer: bool,
        moe_implementation: str,
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

        self.moe = get_moe(
            config,
            moe_implementation=moe_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
        )

        self.mlp = None
        if config.shared_n_inner is not None:
            shared_config = deepcopy(config)
            shared_config.n_inner = config.shared_n_inner
            self.mlp = MLP(shared_config)
            del shared_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        output_router_logits: bool = False,
        output_aux_loss: bool = True,
    ) -> tuple[torch.Tensor]:
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

        hidden_states, router_logits, aux_loss = self._compute_moe_and_mlp(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        # residual connection
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_router_logits:
            outputs += (router_logits,)

        if output_aux_loss:
            outputs += (aux_loss,)

        return outputs