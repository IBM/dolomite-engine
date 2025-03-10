import torch
from transformers import DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .....kernels import wait_for_ACT
from ....enums import AttentionHeadType, PositionEmbeddingType
from ...position_embedding import apply_rotary_pos_emb
from .base import Attention


# Add imports
from .moba.wrapper import moba_layer
from .moba.config import MoBAConfig
from .moba.moba_efficient import moba_attn_varlen



## For the latent attention implementation, we don't need to change anything in the flash attention part!
class FlashAttention2(Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        batch_size, _, query_length, _ = query.shape
        
        # Sparse attention path using MoBA when enabled and in padding-free mode
        if self.use_sparse_attention and cu_seqlens is not None:
            # Prepare inputs for MoBA format (seq_len first)
            # Convert from (batch, heads, seq_len, dim) to (seq_len*batch, heads, dim)
            query_moba = query.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim).contiguous()
            key_moba = key.transpose(1, 2).reshape(-1, self.num_key_value_heads, self.head_dim).contiguous()
            value_moba = value.transpose(1, 2).reshape(-1, self.num_key_value_heads, self.head_dim).contiguous()
            
            # Apply MoBA sparse attention
            hidden_states = moba_attn_varlen(
                q=query_moba,
                k=key_moba,
                v=value_moba,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                moba_chunk_size=self.moba_chunk_size,
                moba_topk=self.moba_topk,
                causal=self.causal,
            )
            
            # Reshape back to expected format: (batch_size, query_length, num_heads*head_dim)
            hidden_states = hidden_states.view(batch_size, query_length, self.num_heads, self.head_dim)
            hidden_states = hidden_states.reshape(batch_size, query_length, -1)
            
        else:
            # Original flash attention path
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

            query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
            key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
            value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

            hidden_states = _flash_attention_forward(
                query_states=query,
                key_states=key,
                value_states=value,
                attention_mask=attention_mask,
                query_length=query_length,
                is_causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                softmax_scale=self._get_softmax_scale(),
            )

            del query, key, value

            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
            hidden_states = hidden_states.view(batch_size, query_length, -1)


        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
