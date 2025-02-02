import torch
from transformers import DynamicCache

from ..moe_dolomite.layer import MoEDolomiteBlock


class MoELadderResidualBlock(MoEDolomiteBlock):
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
        output_aux_loss: bool = True,
    ) -> tuple[torch.Tensor]:
        if previous_attention_out is not None:
            residual = residual + previous_attention_out

        current_attention_out = self.ln_1(residual)
        current_attention_out = self.attn(
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
        current_mlp_out, router_logits, aux_loss = self.moe(current_mlp_out)

        if self.m_residual is not None:
            current_mlp_out = current_mlp_out * self.m_residual

        outputs = (current_attention_out, current_mlp_out, residual)

        if output_aux_loss:
            outputs += (aux_loss,)

        return outputs
