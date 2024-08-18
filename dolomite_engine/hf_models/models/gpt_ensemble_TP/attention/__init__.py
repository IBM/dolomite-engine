from ...gpt_ensemble import GPTEnsembleConfig
from .base import EnsembleAttention_TP
from .flash import EnsembleFlashAttention2_TP
from .sdpa import EnsembleSDPA_TP


_ATTENTION_MODULES = {
    "eager": EnsembleAttention_TP,
    "sdpa": EnsembleSDPA_TP,
    "flash_attention_2": EnsembleFlashAttention2_TP,
}


def get_attention_module(
    config: GPTEnsembleConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> EnsembleAttention_TP:
    if use_padding_free_transformer:
        raise NotImplementedError("padding free transformer is not implemented with tensor parallel")

    if attention_implementation in _ATTENTION_MODULES:
        return _ATTENTION_MODULES[attention_implementation](config, causal=causal, layer_idx=layer_idx)

    raise ValueError(f"unexpected `attention_implementation` {attention_implementation}")