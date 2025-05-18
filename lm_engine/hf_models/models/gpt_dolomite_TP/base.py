from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..gpt_dolomite import GPTDolomiteConfig


class GPTDolomitePreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTDolomiteConfig


class GPTDolomiteModel_TP(GPTDolomitePreTrainedModel_TP, BaseModelMixin_TP): ...
