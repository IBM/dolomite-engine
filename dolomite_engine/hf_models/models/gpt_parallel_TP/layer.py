from torch import Tensor
from transformers import DynamicCache

from ..gpt_dolomite_TP.layer import GPTDolomiteBlock_TP
from ..gpt_parallel.layer import GPTParallelBlock


class GPTParallelBlock_TP(GPTDolomiteBlock_TP):
    def forward(
        self,
        hidden_states: Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: Tensor | None = None,
        rope_cos_sin: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: Tensor | None = None,
    ) -> tuple[Tensor]:
        return GPTParallelBlock.forward(
            self,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
