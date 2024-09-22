import torch
from transformers import DynamicCache

from ..gpt_dolomite.layer import GPTDolomiteBlock


class GPTParallelBlock(GPTDolomiteBlock):
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

        attention_out = self.ln_1(hidden_states)
        attention_out = self.attn(
            attention_out,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        mlp_out = self.ln_2(hidden_states)
        mlp_out = self.mlp(mlp_out)

        hidden_states = attention_out + mlp_out

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        return hidden_states
