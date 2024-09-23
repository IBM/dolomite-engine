import torch
from transformers import DynamicCache

from ..gpt_dolomite_TP.layer import GPTDolomiteBlock_TP
from ..gpt_ladder.layer import GPTLadderBlock


class GPTLadderBlock_TP(GPTDolomiteBlock_TP):
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
        return GPTLadderBlock.forward(
            self,
            previous_attention_out=previous_attention_out,
            previous_mlp_out=previous_mlp_out,
            residual=residual,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
