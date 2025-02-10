from typing import Any

from ...utils import BaseArgs
from ..enums import AttentionHeadType


class _SoftmaxAttentionArgs(BaseArgs):
    sequence_mixer_block_type: str = "softmax_attention"
    num_key_value_heads: int = 1
    attention_head_type: AttentionHeadType = AttentionHeadType.mqa
    softmax_dropout: float
    dropout: float
    add_bias: bool
    attention_multiplier: float | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_block_type == "softmax_attention"
