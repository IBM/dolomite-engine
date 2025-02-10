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


class _Mamba2Args(BaseArgs):
    sequence_mixer_type: str = "mamba2"
    state_size: int = 128
    intermediate_size: int
    num_heads: int = 128
    conv_kernel_size: int = 4
    time_step_rank: int
    time_step_limit: tuple[float, float] = (0, float("inf"))
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    add_bias: bool = True
    use_conv_bias: bool = True
    activation_function: str = "silu"
    num_groups: int = 8
    chunk_size: int = 256

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "mamba2"
