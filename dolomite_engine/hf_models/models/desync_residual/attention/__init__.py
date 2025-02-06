from ..config import DesyncResidualConfig
from .sdpa import DesyncResidualSDPA


_SEQUENCE_MIXERS = {"sdpa": DesyncResidualSDPA}


def get_sequence_mixer(
    config: DesyncResidualConfig,
    causal: bool,
    attention_implementation: str,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> DesyncResidualSDPA:
    if use_padding_free_transformer:
        raise NotImplementedError("padding free transformer is not implemented with DesyncResidual")

    if attention_implementation in _SEQUENCE_MIXERS:
        return _SEQUENCE_MIXERS[attention_implementation](config, causal=causal, layer_idx=layer_idx)

    raise ValueError(f"unexpected `attention_implementation` {attention_implementation}")
