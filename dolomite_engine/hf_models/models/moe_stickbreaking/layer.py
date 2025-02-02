import torch
import torch.nn as nn
from transformers import DynamicCache

from ...modeling_utils import get_normalization_function
from ..moe_dolomite.moe import get_moe
from ..stickbreaking.attention import PaddingFreeSBAttention, SBAttention
from .config import MoEStickBreakingConfig


class MoEStickBreakingBlock(nn.Module):
    def __init__(
        self,
        config: MoEStickBreakingConfig,
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

        if use_padding_free_transformer:
            self.attn = PaddingFreeSBAttention(config, causal=True, layer_idx=layer_idx)
        else:
            self.attn = SBAttention(config, causal=True, layer_idx=layer_idx)

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
        sb_metadata=None,
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
            sb_metadata=sb_metadata,
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

        return hidden_states, aux_loss
