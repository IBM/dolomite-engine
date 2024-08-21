import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from ....enums import PositionEmbeddingType
from ....modeling_utils import apply_rotary_pos_emb
from .base import CrossLayerAttention


class CrossLayerFlashAttention2(CrossLayerAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_length = hidden_states.shape[:2]

        query = self.q_attn(hidden_states)
        query = query.view(batch_size, query_length, self.num_heads, -1)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            # TODO avoid this extra transpose
            query = query.transpose(1, 2)
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            query = query.transpose(1, 2)

        softmax_scale = self._get_softmax_scale()
        dropout_p = self.attn_pdrop if self.training else 0

        batch_size, query_length = query.shape[:2]

        attn_output = _flash_attention_forward(
            query_states=query,
            key_states=key,
            value_states=value,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=self.causal,
            dropout=dropout_p,
            softmax_scale=softmax_scale,
        )

        attn_output = attn_output.view(batch_size, query_length, -1)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output
