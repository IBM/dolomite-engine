from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..gpt_dolomite import GPTDolomiteConfig
from .layer import GPTDolomiteBlock_TP


class GPTDolomitePreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTDolomiteConfig
    layer_class = GPTDolomiteBlock_TP
    _no_split_modules = ["GPTDolomiteBlock_TP"]


class GPTDolomiteModel_TP(GPTDolomitePreTrainedModel_TP, BaseModelMixin_TP): ...
