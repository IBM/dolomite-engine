import torch
import torch.nn as nn
from transformers import DynamicCache

from ...modeling_utils import get_attention_module, get_normalization_function
from ..gpt_dolomite.mlp import MLP
from .config import GPTParallelConfig


class GPTParallelBlock(nn.Module):
    def __init__(
        self,
        config: GPTParallelConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln = get_normalization_function(config.normalization_function, hidden_size, eps=config.layer_norm_epsilon)
        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
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
    ) -> tuple[torch.Tensor]:
        residual = hidden_states

        hidden_states = self.ln(hidden_states)

        # NOTE we can contenate the input matrices of attention and MLP here for speedup
        # but right now we avoid it since this code is only used for accuracy benchmarking at small scale
        attention_out = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        mlp_out = self.mlp(hidden_states)

        hidden_states = attention_out + mlp_out

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        return hidden_states
