from ...mixins import BaseMoEModelMixin, PreTrainedMoEModelMixin
from .config import MoEDolomiteConfig
from .layer import MoEDolomiteBlock


class MoEDolomitePreTrainedModel(PreTrainedMoEModelMixin):
    config_class = MoEDolomiteConfig
    layer_class = MoEDolomiteBlock
    _no_split_modules = ["MoEDolomiteBlock"]


class MoEDolomiteModel(MoEDolomitePreTrainedModel, BaseMoEModelMixin): ...
