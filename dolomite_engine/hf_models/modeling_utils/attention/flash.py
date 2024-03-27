from typing import Tuple

import torch

from ...enums import AttentionHeadType, PositionEmbeddingType
from ..position_embedding import apply_rotary_pos_emb
from .base import Attention
from .utils import unpad_tensor


try:
    from flash_attn.bert_padding import IndexFirstAxis, pad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
    pad_input = None
    IndexFirstAxis = None


class FlashAttention2(Attention):
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
        assert alibi_bias is None
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

        # TODO avoid this extra transpose
        query = query.transpose(1, 2)
        if self.attention_head_type == AttentionHeadType.mqa:
            key = key.squeeze(1).unsqueeze(2)
            value = value.squeeze(1).unsqueeze(2)
        else:
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        # ==========================================================================================
        # query -> (batch_size, query_length, num_heads, head_dim)
        # key -> (batch_size, key_length, num_heads, head_dim)
        # value -> (batch_size, key_length, num_heads, head_dim)
        # ==========================================================================================

        batch_size, query_length = query.shape[:2]
        key_length = key.shape[1]
        _, indices_k, cu_seqlens_k, max_seqlen_k = unpad_tensor(None, attention_mask)

        key = IndexFirstAxis.apply(
            key.reshape(batch_size * key_length, self.num_key_value_heads, self.head_dim), indices_k
        )
        value = IndexFirstAxis.apply(
            value.reshape(batch_size * key_length, self.num_key_value_heads, self.head_dim), indices_k
        )

        if query_length == key_length:
            query = IndexFirstAxis.apply(
                query.reshape(batch_size * key_length, self.num_heads, self.head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query = query.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query, indices_q, cu_seqlens_q, max_seqlen_q = unpad_tensor(query, attention_mask)

        # ==========================================================================================
        # query -> (total_q, num_heads, head_dim)
        # key -> (total_q, num_heads, head_dim)
        # value -> (total_q, num_heads, head_dim)
        # ==========================================================================================

        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=self.attn_pdrop if self.training else 0,
            softmax_scale=None if self.scale_attn_weights else 1,
            causal=self.causal,
        )

        # ==========================================================================================
        # attn_output -> (total_q, num_heads, head_dim)
        # ==========================================================================================

        attn_output = pad_input(attn_output, indices_q, batch_size, query_length)
        attn_output = attn_output.view(batch_size, query_length, -1)

        # ==========================================================================================
        # attn_output -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present
