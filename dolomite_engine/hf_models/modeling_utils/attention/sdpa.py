from typing import Tuple

import torch
import torch.nn.functional as F

from ...enums import PositionEmbeddingType
from ..position_embedding import apply_rotary_pos_emb
from .base import Attention
from .utils import repeat_key_value


class SDPA(Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # alibi_bias is already added in base model to the attention_mask
        assert not output_attentions

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=2)
            value = torch.cat((layer_past[1], value), dim=2)

        present = (key, value) if use_cache else None

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_heads, key_length, head_dim)
        # value -> (batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_pdrop if self.training else 0,
            is_causal=self.causal if attention_mask is None else False,
            scale=None if self.scale_attn_weights else 1,
        )

        # ==========================================================================================
        # attn_output -> (batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        batch_size = attn_output.shape[0]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        # ==========================================================================================
        # attn_output -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present
