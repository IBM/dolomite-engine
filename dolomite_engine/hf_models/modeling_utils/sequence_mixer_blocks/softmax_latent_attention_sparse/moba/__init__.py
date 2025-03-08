from functools import partial
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .wrapper import moba_layer
from .moba_naive import moba_attn_varlen_naive
from .moba_efficient import moba_attn_varlen
from .config import MoBAConfig


def register_moba(cfg: MoBAConfig):
    ALL_ATTENTION_FUNCTIONS["moba_naive"] = partial(moba_layer, moba_attn_varlen_naive, cfg)
    ALL_ATTENTION_FUNCTIONS["moba"] = partial(moba_layer, moba_attn_varlen, cfg)
