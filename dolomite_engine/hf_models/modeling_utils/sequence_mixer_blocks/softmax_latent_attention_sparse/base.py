import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from .....utils import divide_if_divisible
from ....enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ...linear import ParameterizedLinear
from ...position_embedding import apply_rotary_pos_emb
from ..softmax_attention import repeat_key_value
from .moba.config import MoBAConfig

# Import needed for MoBA
from .moba.wrapper import moba_layer


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: InitMethod,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        kv_compression_dim: int = None,
        head_dim_latent: int = 64,
        use_latent_attention: bool = False,
        use_sparse_attention: bool = False,
        sparse_pattern: str = "block_local",
        moba_chunk_size: int = 1024,
        moba_topk: int = 8,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias

        self.kv_compression_dim = kv_compression_dim
        self.use_latent_attention = use_latent_attention

        # Sparse attention params
        self.use_sparse_attention = use_sparse_attention
        self.sparse_pattern = sparse_pattern
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk

        if not use_latent_attention:
            ## For the latent attention work, we don't need to check this division
            self.head_dim = divide_if_divisible(
                self.hidden_size,
                self.num_heads,
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
            )
        else:
            # Just hardcode here, need to pass dimenstion later.
            self.head_dim = head_dim_latent

        self.attention_head_type = attention_head_type
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        if self.attention_head_type == AttentionHeadType.mha:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.num_heads

            assert (
                self.num_heads == self.num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"
        elif self.attention_head_type == AttentionHeadType.gqa:
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            divide_if_divisible(
                self.num_heads,
                self.num_key_value_heads,
                f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` ({self.num_key_value_heads})",
            )
        elif self.attention_head_type == AttentionHeadType.mqa:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, f"{self.__class__.__name__} should have 1 head for keys and values"
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        # add the latent attention layer here
        if self.use_latent_attention:
            assert (
                self.kv_compression_dim is not None
            ), "kv_compression_dim must be specified when using latent attention"
            # For latent attention, we have separate projections
            self.q_proj = ParameterizedLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )

            self.kv_down_proj = ParameterizedLinear(
                self.hidden_size,
                self.kv_compression_dim,
                bias=self.add_bias,
                std=std,
            )

            self.k_up_proj = ParameterizedLinear(
                self.kv_compression_dim,
                self.num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )

            self.v_up_proj = ParameterizedLinear(
                self.kv_compression_dim,
                self.num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )
        else:
            # Original implementation for non-latent attention
            self.c_attn = ParameterizedLinear(
                self.hidden_size,
                self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
            )

        # Add to the __init__ method in Attention class
        if self.use_sparse_attention:
            # Register MoBA with standard configurations
            from .moba import register_moba
            from .moba.config import MoBAConfig

            # Configure MoBA with user parameters
            moba_config = MoBAConfig(
                moba_chunk_size=self.moba_chunk_size,  # Sensible default based on block size
                moba_topk=self.moba_topk,  # Sensible default based on sparsity level
            )

            # Register MoBA implementations
            register_moba(moba_config)

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=self.add_bias, std=std
        )

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

    ## TODO: Need to verify if the current implementation is correct or not.
    def _prepare_qkv_for_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        if self.use_latent_attention:
            # Latent attention flow
            query = self.q_proj(hidden_states)
            kv_latent = self.kv_down_proj(hidden_states)
            key = self.k_up_proj(kv_latent)
            value = self.v_up_proj(kv_latent)

            batch_size, query_length = hidden_states.shape[:-1]

            # Reshape query
            query = query.view(batch_size, query_length, self.num_heads, self.head_dim)
            query = query.transpose(1, 2)

            # Reshape key and value based on attention head type
            if self.attention_head_type == AttentionHeadType.mha:
                key = key.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
                value = value.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
            else:
                raise ValueError(
                    f"unexpected attention_head_type ({self.attention_head_type}). \
                                   The only supported attention_head_type is mha"
                )
        else:
            # Original attention flow
            hidden_states = self.c_attn(hidden_states)

            # Use existing methods based on attention head type
            if self.attention_head_type == AttentionHeadType.mha:
                query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
            elif self.attention_head_type == AttentionHeadType.gqa:
                query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
            elif self.attention_head_type == AttentionHeadType.mqa:
                query, key, value = self._prepare_qkv_for_forward_mqa(hidden_states)
            else:
                raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================
        return query, key, value

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_heads, -1)
        hidden_states = hidden_states.transpose(1, 2)

        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_key_value_heads, -1)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(batch_size, query_length, -1, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        query, key, value = hidden_states.split((self.hidden_size, self.head_dim, self.head_dim), dim=-1)

        query = query.view(batch_size, query_length, self.num_heads, -1)

        query = query.transpose(1, 2)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value

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

        # Check if sparse attention should be used
        if self.use_sparse_attention and cu_seqlens is not None:
            # Configure MoBA
            moba_config = MoBAConfig(
                moba_chunk_size=self.moba_chunk_size,  # Sensible default based on block size
                moba_topk=self.moba_topk,  # Sensible default based on sparsity level
            )

            # Use MoBA for sparse attention
            from .moba.moba_efficient import moba_attn_varlen

            # Convert from batched to varlen format for MoBA
            # MoBA expects seq_len, head, head_dim format
            batch_size, num_heads, query_length, head_dim = query.shape
            reshaped_query = query.permute(2, 1, 3, 0).reshape(query_length * batch_size, num_heads, head_dim)
            reshaped_key = key.permute(2, 1, 3, 0).reshape(
                key.shape[2] * batch_size, self.num_key_value_heads, head_dim
            )
            reshaped_value = value.permute(2, 1, 3, 0).reshape(
                value.shape[2] * batch_size, self.num_key_value_heads, head_dim
            )

            # Adjust cu_seqlens for multiple batches
            if batch_size > 1:
                batch_offsets = torch.arange(0, batch_size, device=query.device) * query_length
                adjusted_cu_seqlens = torch.cat([cu_seqlens + offset for offset in batch_offsets])
            else:
                adjusted_cu_seqlens = cu_seqlens

            # Call MoBA attention
            output = moba_attn_varlen(
                q=reshaped_query,
                k=reshaped_key,
                v=reshaped_value,
                cu_seqlens=adjusted_cu_seqlens,
                max_seqlen=max_seqlen,
                moba_chunk_size=self.moba_chunk_size,  # Configurable chunk size
                moba_topk=self.moba_topk,  # Configurable sparsity level
            )

            # Reshape back to expected format
            output = output.view(query_length, batch_size, num_heads, head_dim)
            output = output.permute(1, 2, 0, 3)
            hidden_states = output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

        else:

            # ==========================================================================================
            # query -> (batch_size, num_heads, query_length, head_dim)
            # key -> (batch_size, num_key_value_heads, key_length, head_dim)
            # value -> (batch_size, num_key_value_heads, key_length, head_dim)
            # ==========================================================================================

            key = key.transpose(-1, -2)
            dtype = query.dtype

            # ==========================================================================================
            # query -> (batch_size, num_heads, query_length, head_dim)
            # key -> (batch_size, num_key_value_heads, head_dim, key_length)
            # value -> (batch_size, num_key_value_heads, key_length, head_dim)
            # ==========================================================================================

            batch_size = query.shape[0]
            query_length = query.shape[2]
            key_length = key.shape[-1]

            # if the #group = 1, actually no return.
            key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
            value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

            # Always copies
            query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
            # No copy when layer_past is provided.
            key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

            # ==========================================================================================
            # query -> (batch_size * num_heads, query_length, head_dim)
            # key -> (batch_size * num_heads, head_dim, key_length)
            # value -> (batch_size, num_heads, key_length, head_dim)
            # ==========================================================================================

            if attention_mask is None:
                hidden_states = torch.empty(
                    (batch_size * self.num_heads, query_length, key_length), device=query.device, dtype=query.dtype
                )
                beta = 0
            else:
                hidden_states = attention_mask.expand(-1, self.num_heads, -1, -1).reshape(-1, query_length, key_length)
                beta = 1

            hidden_states = torch.baddbmm(
                hidden_states, query, key, beta=beta, alpha=self._get_softmax_scale(False)
            ).view(batch_size, self.num_heads, query_length, key_length)

            del query, key

            # ==========================================================================================
            # hidden_states -> (batch_size, num_heads, query_length, key_length)
            # ==========================================================================================

            hidden_states = F.softmax(hidden_states.float(), dim=-1).to(dtype)
            hidden_states = self.softmax_dropout(hidden_states)

            # ==========================================================================================
            # value -> (batch_size, num_heads, key_length, head_dim)
            # hidden_states -> (batch_size, num_heads, query_length, key_length)
            # ==========================================================================================

        hidden_states = torch.matmul(hidden_states, value)

        del value

        # ==========================================================================================
        # hidden_states -> (batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def _get_softmax_scale(self, return_none_allowed: bool = True) -> float:
        if self.attention_multiplier is None:
            softmax_scale = None if return_none_allowed else 1 / self.head_dim**0.5
        else:
            softmax_scale = self.attention_multiplier

        return softmax_scale
