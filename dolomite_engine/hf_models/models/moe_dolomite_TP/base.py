from ...mixins import BaseMoEModelMixin_TP, PreTrainedModelMixin_TP
from .layer import MoEDolomiteBlock_TP


class MoEDolomitePreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = None
    layer_class = MoEDolomiteBlock_TP
    _no_split_modules = ["MoEDolomiteBlock_TP"]


class MoEDolomiteModel_TP(MoEDolomitePreTrainedModel_TP, BaseMoEModelMixin_TP): ...
