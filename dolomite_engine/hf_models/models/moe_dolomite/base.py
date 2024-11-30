from ...mixins import BaseMoEModelMixin, PreTrainedMoEModelMixin
from .config import MoEDolomiteConfig
from .layer import SparseMoEBlock


class MoEDolomitePreTrainedModel(PreTrainedMoEModelMixin):
    config_class = MoEDolomiteConfig
    layer_class = SparseMoEBlock
    _no_split_modules = ["SparseMoEBlock"]


class MoEDolomiteModel(MoEDolomitePreTrainedModel, BaseMoEModelMixin): ...
