import torch

from ..config import CommonConfig
from .softmax_attention import _SoftmaxAttentionCache


class _Mamba2Cache(_SoftmaxAttentionCache):
    def __init__(self, config: CommonConfig, layer_idx: int, **kwargs) -> None:
        self.seen_tokens = 0
        self.conv_cache: torch.Tensor | None = None
        self.ssm_cache: torch.Tensor | None = None

    def get_cache(self) -> tuple[torch.Tensor | None]:
        return self.conv_cache, self.ssm_cache

    def update(
        self,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        num_tokens_added: int = 1,
        sequence_length_dimension: int = -2,
    ) -> tuple[torch.Tensor]:
        self.seen_tokens += num_tokens_added
        self.conv_cache = conv_state
        self.ssm_cache = ssm_state
        return self.conv_cache, self.ssm_cache

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self.conv_cache = self.conv_cache.index_select(0, beam_idx.to(self.conv_cache.device))
        self.ssm_cache = self.ssm_cache.index_select(0, beam_idx.to(self.ssm_cache.device))
