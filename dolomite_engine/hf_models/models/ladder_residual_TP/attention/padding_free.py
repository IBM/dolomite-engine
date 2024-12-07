import torch
from transformers import DynamicCache

from .....utils import is_flash_attention_available
from ....enums import PositionEmbeddingType
from ....modeling_utils import apply_rotary_pos_emb
from ....modeling_utils_TP.attention import PaddingFreeAttention_TP


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class LadderResidualPaddingFreeAttention_TP(PaddingFreeAttention_TP):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert past_key_values is None

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        softmax_scale = self._get_softmax_scale()
        dropout_p = self.attn_pdrop if self.training else 0

        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=self.causal,
        )

        attn_output = attn_output.view(-1, self.hidden_size)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output
