import torch
import torch.nn as nn
from transformers import DynamicCache

from ...modeling_utils import get_attention_module, get_normalization_function
from .config import MoEDolomiteConfig
from .moe import get_moe


class MoEDolomiteBlock(nn.Module):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        moe_implementation: str,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

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
        self.moe = get_moe(
            config,
            moe_implementation=moe_implementation,
            use_aux_free_moe=config.use_aux_free_moe,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
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
            attention_mask=attention_mask,
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

        hidden_states, router_logits, aux_loss = self.moe(hidden_states)

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
