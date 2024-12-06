import torch
import torch.nn as nn
from transformers import DynamicCache

from ...enums import AttentionHeadType
from ...modeling_utils import get_normalization_function
from ..gpt_dolomite.mlp import MLP
from ..stickbreaking.attention import PaddingFreeSBAttention, SBAttention
from .config import MoEStickBreakingConfig


class MoEStickBreakingBlock(nn.Module):
    def __init__(
        self,
        config: MoEStickBreakingConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
        )

        if use_padding_free_transformer:
            self.attn = PaddingFreeSBAttention(config, causal=True, layer_idx=layer_idx)
        else:
            self.attn = SBAttention(config, causal=True, layer_idx=layer_idx)

        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.mlp = MLP(config)

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

        attn_output = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sb_metadata=sb_metadata,
        )

        if self.m_residual is not None:
            attn_output = attn_output * self.m_residual

        # residual connection
        hidden_states = attn_output + residual
        residual = hidden_states

        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)

        if self.m_residual is not None:
            feed_forward_hidden_states = feed_forward_hidden_states * self.m_residual

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states
