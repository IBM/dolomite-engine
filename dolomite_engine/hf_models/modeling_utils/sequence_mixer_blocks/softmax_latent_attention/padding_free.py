import torch
from transformers import DynamicCache

from ..softmax_attention import PaddingFreeAttention
from .base import MLAAttention


class MLAPaddingFreeAttention(MLAAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return PaddingFreeAttention.forward(
            self,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return PaddingFreeAttention._prepare_qkv_for_forward_mha(self, hidden_states=hidden_states)

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return PaddingFreeAttention._prepare_qkv_for_forward_gqa(self, hidden_states=hidden_states)

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return PaddingFreeAttention._prepare_qkv_for_forward_mqa(self, hidden_states=hidden_states)
