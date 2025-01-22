from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import MambaDolomiteConfig
from .layer import MambaDolomiteBlock


class MambaDolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = MambaDolomiteConfig
    layer_class = MambaDolomiteBlock
    _no_split_modules = ["MambaDolomiteBlock"]


class MambaDolomiteModel(MambaDolomitePreTrainedModel, BaseModelMixin): ...
