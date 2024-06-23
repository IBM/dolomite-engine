import torch
from transformers import DynamicCache

from dolomite_engine.hf_models.config import CommonConfig

from ....modeling_utils import PaddingFreeAttention
from .base import EnsembleAttention


class EnsemblePaddingFreeAttention(PaddingFreeAttention):
    def __init__(self, config: CommonConfig, causal: bool, layer_idx: int = None) -> None:
        EnsembleAttention.__init__(self, config, causal, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache = None,
        attention_mask: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> torch.Tensor:
        total_q, pretraining_tp, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(total_q * pretraining_tp, hidden_size)
        attention_output = super().forward(
            hidden_states, past_key_values, attention_mask, rope_cos_sin, cu_seqlens, max_seqlen
        )
        attention_output = attention_output.view(total_q, pretraining_tp, hidden_size)
        return attention_output
