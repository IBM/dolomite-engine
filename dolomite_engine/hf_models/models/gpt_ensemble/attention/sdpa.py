import torch
import torch.nn.functional as F
from transformers import DynamicCache

from ....enums import PositionEmbeddingType
from ....modeling_utils import apply_rotary_pos_emb, repeat_key_value
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
        # ==========================================================================================
        # hidden_states -> (1, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (TP * batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_heads, key_length, head_dim)
        # value -> (TP * batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

        softmax_scale = self._get_softmax_scale()
        dropout_p = self.attn_pdrop if self.training else 0

        if attention_mask is not None:
            # TODO avoid this repeat on every layer
            attention_mask = attention_mask.repeat(self.tp_world_size, 1, 1, 1)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=self.causal if attention_mask is None else False,
            scale=softmax_scale,
        )

        # ==========================================================================================
        # attn_output -> (TP * batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        query_length = attn_output.shape[2]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(self.tp_world_size, -1, query_length, self.num_heads * self.head_dim)

        # ==========================================================================================
        # attn_output -> (TP, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output
