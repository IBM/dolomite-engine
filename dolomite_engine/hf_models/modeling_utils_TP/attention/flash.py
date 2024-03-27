from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import FlashAttention2
from .base import Attention_TP


class FlashAttention2_TP(Attention_TP, FlashAttention2):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        causal: bool,
        add_bias: bool,
        scale_attention_weights: bool,
        attention_softmax_in_fp32: bool,
        scale_attention_softmax_in_fp32: bool,
        attn_pdrop: float,
        resid_pdrop: float,
        layer_idx: int = None,
    ) -> None:
        Attention_TP.__init__(
            self,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            attention_head_type,
            position_embedding_type,
            causal,
            add_bias,
            scale_attention_weights,
            attention_softmax_in_fp32,
            scale_attention_softmax_in_fp32,
            attn_pdrop,
            resid_pdrop,
            layer_idx,
        )
