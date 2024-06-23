from ..config import GPTEnsembleConfig
from .base import EnsembleAttention
from .sdpa import EnsembleSDPA


_ATTENTION_MODULES = {
    "eager": EnsembleAttention,
    "sdpa": EnsembleSDPA,
}


def get_attention_module(
    config: GPTEnsembleConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> EnsembleAttention:
    if use_padding_free_transformer:
        raise NotImplementedError("padding free transformer is not implemented with GPTEnsemble")

    if attention_implementation in _ATTENTION_MODULES:
        return _ATTENTION_MODULES[attention_implementation](config, causal=causal, layer_idx=layer_idx)

    raise ValueError(f"unexpected `attention_implementation` {attention_implementation}")
