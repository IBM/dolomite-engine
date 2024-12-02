from ...desync_residual import DesyncResidualConfig
from .sdpa import DesyncResidualSDPA_TP


_ATTENTION_MODULES = {"sdpa": DesyncResidualSDPA_TP}


def get_attention_module(
    config: DesyncResidualConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> DesyncResidualSDPA_TP:
    if use_padding_free_transformer:
        raise NotImplementedError("padding free transformer is not implemented with tensor parallel")

    if attention_implementation in _ATTENTION_MODULES:
        return _ATTENTION_MODULES[attention_implementation](config, causal=causal, layer_idx=layer_idx)

    raise ValueError(f"unexpected `attention_implementation` {attention_implementation}")
