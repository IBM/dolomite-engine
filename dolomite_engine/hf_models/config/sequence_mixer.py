from typing import Any

from ...utils import BaseArgs
from ..enums import AttentionHeadType


class _SoftmaxAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "softmax_attention"
    num_key_value_heads: int = 1
    attention_head_type: AttentionHeadType = AttentionHeadType.mqa
    softmax_dropout: float = 0
    dropout: float = 0
    add_bias: bool = True
    attention_multiplier: float | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "softmax_attention"
