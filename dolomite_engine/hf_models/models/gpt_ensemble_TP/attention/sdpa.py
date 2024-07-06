from ....modeling_utils import SDPA
from ...gpt_ensemble import GPTEnsembleConfig
from .base import EnsembleAttention_TP


class EnsembleSDPA_TP(EnsembleAttention_TP, SDPA):
    def __init__(self, config: GPTEnsembleConfig, causal: bool, layer_idx: int = None) -> None:
        EnsembleAttention_TP.__init__(self, config=config, causal=causal, layer_idx=layer_idx)
