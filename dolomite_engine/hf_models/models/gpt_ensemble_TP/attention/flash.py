from ....modeling_utils import FlashAttention2
from ...gpt_ensemble import GPTEnsembleConfig
from .base import EnsembleAttention_TP


class EnsembleFlashAttention2_TP(EnsembleAttention_TP, FlashAttention2):
    def __init__(self, config: GPTEnsembleConfig, causal: bool, layer_idx: int = None) -> None:
        EnsembleAttention_TP.__init__(self, config=config, causal=causal, layer_idx=layer_idx)
