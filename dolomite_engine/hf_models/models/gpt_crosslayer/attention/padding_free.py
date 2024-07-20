import torch

from .....utils import is_flash_attention_available
from ....enums import PositionEmbeddingType
from ....modeling_utils import apply_rotary_pos_emb
from .base import CrossLayerAttention, KeyValueProjection


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class CrossLayerPaddingFreeAttention(CrossLayerAttention):
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
        total_q = hidden_states.shape[0]

        query = self.q_attn(hidden_states)
        query = query.view(total_q, self.num_heads, -1)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)

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


class KeyValuePaddingFreeProjection(KeyValueProjection):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln(hidden_states)
        hidden_states = self.kv_attn(hidden_states)

        if self.num_key_value_heads == 1:
            hidden_states = hidden_states.unsqueeze(1)
        else:
            total_q = hidden_states.shape[0]
            hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)

        key, value = hidden_states.chunk(2, -1)

        return key, value
