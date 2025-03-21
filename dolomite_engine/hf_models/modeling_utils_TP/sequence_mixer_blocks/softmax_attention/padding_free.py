import torch

from ....enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from .base import _BaseAttention_TP


class PaddingFreeAttention_TP(_BaseAttention_TP):
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
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        self._init_attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_multiplier=attention_multiplier,
            attention_head_type=attention_head_type,
            position_embedding_type=position_embedding_type,
            add_bias=add_bias,
            softmax_dropout=softmax_dropout,
            dropout=dropout,
            init_method=init_method,
            initializer_range=initializer_range,
            m_width=m_width,
            num_layers=num_layers,
            causal=causal,
            layer_idx=layer_idx,
            use_padding_free_transformer=True,
            sequence_parallel=sequence_parallel,
        )

    def _prepare_qkv_for_forward_mqa(
        self, query_key_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = query_key_value

        total_q = query.shape[0]

        query = query.view(total_q, self.num_heads, -1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value
