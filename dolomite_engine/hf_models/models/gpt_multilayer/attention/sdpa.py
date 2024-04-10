import torch
import torch.nn.functional as F

from ....enums import PositionEmbeddingType
from ....modeling_utils import apply_rotary_pos_emb
from .base import MultiLayerAttention


class MultiLayerSDPA(MultiLayerAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, query_length = hidden_states.shape[:2]

        query = self.q_attn(hidden_states)
        query = query.view(batch_size, query_length, self.num_heads, -1)
        query = query.transpose(1, 2)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_pdrop if self.training else 0,
            is_causal=self.causal if attention_mask is None else False,
            scale=self.attention_multiplier if self.scale_attn_weights else 1,
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output
