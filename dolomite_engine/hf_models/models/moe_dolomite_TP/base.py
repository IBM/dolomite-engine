from ...mixins import BaseMoEModelMixin_TP, PreTrainedMoEModelMixin_TP
from ..moe_dolomite import MoEDolomiteConfig
from .layer import SparseMoEBlock_TP


class MoEDolomitePreTrainedModel_TP(PreTrainedMoEModelMixin_TP):
    config_class = MoEDolomiteConfig
    layer_class = SparseMoEBlock_TP
    _no_split_modules = ["SparseMoEBlock_TP"]


class MoEDolomiteModel_TP(MoEDolomitePreTrainedModel_TP, BaseMoEModelMixin_TP): ...
