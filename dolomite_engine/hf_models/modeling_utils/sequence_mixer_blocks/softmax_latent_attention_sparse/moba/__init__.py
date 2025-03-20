from functools import partial

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .config import MoBAConfig
from .moba_efficient import moba_attn_varlen
from .moba_naive import moba_attn_varlen_naive
from .wrapper import moba_layer


def register_moba(cfg: MoBAConfig):
    ALL_ATTENTION_FUNCTIONS["moba_naive"] = partial(moba_layer, moba_attn_varlen_naive, cfg)
    ALL_ATTENTION_FUNCTIONS["moba"] = partial(moba_layer, moba_attn_varlen, cfg)
