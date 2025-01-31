from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import Mamba2DolomiteConfig
from .layer import Mamba2DolomiteBlock


class Mamba2DolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = Mamba2DolomiteConfig
    layer_class = Mamba2DolomiteBlock
    _no_split_modules = ["Mamba2DolomiteBlock"]


class Mamba2DolomiteModel(Mamba2DolomitePreTrainedModel, BaseModelMixin): ...
