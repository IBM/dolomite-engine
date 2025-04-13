from typing import Any

from ...utils import BaseArgs


class _SoftmaxAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "softmax_attention"
    num_query_heads: int = 12
    num_key_value_heads: int = 1
    softmax_dropout: float = 0
    dropout: float = 0
    add_bias: bool = True
    attention_multiplier: float | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "softmax_attention"


class _MultiHeadLatentAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "multihead_latent_attention"
    softmax_dropout: float = 0
    dropout: float = 0
    add_bias: bool = True
    attention_multiplier: float | None = None
    query_compression_size: int | None = None
    key_value_compression_size: int | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "multihead_latent_attention"
        assert self.query_compression_size is not None
        assert self.key_value_compression_size is not None


class _StickbreakingAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "stickbreaking_attention"
    num_query_heads: int = 12
    num_key_value_heads: int = 1
    dropout: float = 0
    add_bias: bool = True
    attention_multiplier: float | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "stickbreaking_attention"


class _Mamba2Args(BaseArgs):
    sequence_mixer_type: str = "mamba2"
    state_size: int = 128
    intermediate_size: int
    num_heads: int = 128
    conv_kernel_size: int = 4
    time_step_limit: tuple[float, float] = (0, float("inf"))
    add_bias: bool = True
    use_conv_bias: bool = True
    activation_function: str = "silu"
    num_groups: int = 8
    chunk_size: int = 256

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "mamba2"


class _RNNArgs(BaseArgs):
    sequence_mixer_type: str = "rnn"
    state_size: int = 2048
    num_heads: int = 128
    add_bias: bool = True

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "rnn"
