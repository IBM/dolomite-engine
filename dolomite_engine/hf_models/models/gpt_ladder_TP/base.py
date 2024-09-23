from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..gpt_ladder import GPTLadderConfig
from .layer import GPTLadderBlock_TP


class GPTLadderPreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTLadderConfig
    layer_class = GPTLadderBlock_TP
    _no_split_modules = ["GPTLadderBlock_TP"]


class GPTLadderModel_TP(GPTLadderPreTrainedModel_TP, BaseModelMixin_TP): ...
