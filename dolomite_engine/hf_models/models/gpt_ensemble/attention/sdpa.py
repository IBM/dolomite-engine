import torch
from transformers import DynamicCache

from ....modeling_utils import SDPA
from .base import EnsembleAttention


class EnsembleSDPA(EnsembleAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache = None,
        attention_mask: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, pretraining_tp, query_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * pretraining_tp, query_length, hidden_size)
        attention_output = SDPA.forward(
            self, hidden_states, past_key_values, attention_mask, rope_cos_sin, cu_seqlens, max_seqlen
        )
        attention_output = attention_output.view(batch_size, pretraining_tp, query_length, hidden_size)
        return attention_output
