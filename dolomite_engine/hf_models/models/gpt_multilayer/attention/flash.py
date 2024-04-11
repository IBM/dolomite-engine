import torch

from .....utils import is_flash_attention_available
from ....enums import PositionEmbeddingType
from ....modeling_utils import apply_rotary_pos_emb, unpad_tensor
from .base import MultiLayerAttention


if is_flash_attention_available():
    from flash_attn.bert_padding import IndexFirstAxis, pad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class MultiLayerFlashAttention2(MultiLayerAttention):
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
        assert alibi_bias is None

        batch_size, query_length = hidden_states.shape[:2]

        query = self.q_attn(hidden_states)
        query = query.view(batch_size, query_length, self.num_heads, -1)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            # TODO avoid this extra transpose
            query = query.transpose(1, 2)
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            query = query.transpose(1, 2)

        batch_size, query_length = query.shape[:2]
        key_length = key.shape[1]
        _, indices_k, cu_seqlens_k, max_seqlen_k = unpad_tensor(None, attention_mask)

        # TODO: figure out a way to move this outside
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

        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=self.attn_pdrop if self.training else 0,
            softmax_scale=self.attention_multiplier if self.scale_attn_weights else 1,
            causal=self.causal,
        )
        attn_output = pad_input(attn_output, indices_q, batch_size, query_length)
        attn_output = attn_output.view(batch_size, query_length, -1)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output
