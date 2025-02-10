import torch
from transformers import DynamicCache

from ..gpt_dolomite.layer import GPTDolomiteBlock


class LadderResidualBlock(GPTDolomiteBlock):
    def forward(
        self,
        previous_attention_out: torch.Tensor,
        previous_mlp_out: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if previous_attention_out is not None:
            residual = residual + previous_attention_out

        current_attention_out = self.ln_1(residual)
        current_attention_out = self.sequence_mixer(
            current_attention_out,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.m_residual is not None:
            current_attention_out = current_attention_out * self.m_residual

        if previous_mlp_out is not None:
            residual = residual + previous_mlp_out

        current_mlp_out = self.ln_2(residual)
        current_mlp_out = self.mlp_block(current_mlp_out)

        if self.m_residual is not None:
            current_mlp_out = current_mlp_out * self.m_residual

        return current_attention_out, current_mlp_out, residual
