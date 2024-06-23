from ..config import GPTEnsembleConfig
from .base import EnsembleAttention
from .flash import EnsembleFlashAttention2

# from .padding_free import EnsemblePaddingFreeAttention
from .sdpa import EnsembleSDPA


_ATTENTION_MODULES = {
    "eager": EnsembleAttention,
    "sdpa": EnsembleSDPA,
    "flash_attention_2": EnsembleFlashAttention2,
}


def get_attention_module(
    config: GPTEnsembleConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> EnsembleAttention:
    if use_padding_free_transformer:
        assert (
            attention_implementation == "flash_attention_2"
        ), "padding free transformer only works with flash attention"
        attention_class = EnsemblePaddingFreeAttention
    else:
        attention_class = _ATTENTION_MODULES[attention_implementation]

    return attention_class(config, causal=causal, layer_idx=layer_idx)
