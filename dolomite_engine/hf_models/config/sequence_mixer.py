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
    
    
    # Add new parameters for latent attention
    use_latent_attention: bool = False
    kv_compression_dim: int | None = None
    head_dim_latent: int | None = None
    
    # Add new parameters for sparse attention
    use_sparse_attention: bool = False
    sparse_block_size: int = 16
    sparse_pattern: str = "block_local"
    moba_chunk_size: int = 1024
    moba_topk: int = 8


    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "softmax_attention"
                # Add validation for latent and sparse attention
        if self.use_latent_attention:
            assert self.kv_compression_dim is not None, "kv_compression_dim must be set when use_latent_attention is True"
            assert self.head_dim_latent is not None, "head_dim_latent must be set when use_latent_attention is True"
        if self.use_sparse_attention:
            assert self.sparse_pattern in ["block_local", "block_sparse", "strided"], f"Invalid sparse_pattern: {self.sparse_pattern}"

class _StickbreakingAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "stickbreaking_attention"
    num_key_value_heads: int = 1
    attention_head_type: AttentionHeadType = AttentionHeadType.mqa
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
