import torch
from transformers import DynamicCache

from ..gpt_dolomite_TP.layer import GPTDolomiteBlock_TP
from .rmsnorm import rmsnorm_cute_backward_only, rmsnorm_cute_forward_only


class LadderResidualBlock_TP(GPTDolomiteBlock_TP):
    def forward(
        self,
        current_attention_out: torch.Tensor,
        current_mlp_out: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if current_attention_out is not None:
            residual = residual + current_attention_out

        ln_1_weight = self.ln_1.weight
        (
            current_attention_out,
            current_attention_x,
            ln_1_weight,
            current_attention_rmsnorm_denominator,
            attention_is_x_1d,
        ) = rmsnorm_cute_forward_only(residual, ln_1_weight, self.ln_1.eps, False)

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

        if current_mlp_out is not None:
            residual = residual + current_mlp_out

        ln_2_weight = self.ln_2.weight
        current_mlp_out, current_mlp_x, ln_2_weight, current_mlp_rmsnorm_denominator, mlp_is_x_1d = (
            rmsnorm_cute_forward_only(residual, ln_2_weight, self.ln_2.eps, False)
        )

        current_mlp_out = self.mlp_block(current_mlp_out)

        if self.m_residual is not None:
            current_mlp_out = current_mlp_out * self.m_residual

        return current_attention_out, current_mlp_out, residual
