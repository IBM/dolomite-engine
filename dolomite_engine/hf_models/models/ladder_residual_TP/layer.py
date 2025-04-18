import torch
from transformers import DynamicCache

from ...mixins import BaseBlock_TP
from ..ladder_residual.layer import LadderResidualBlock


class LadderResidualBlock_TP(BaseBlock_TP):
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
        return LadderResidualBlock.forward(
            self,
            current_attention_out=current_attention_out,
            current_mlp_out=current_mlp_out,
            residual=residual,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
