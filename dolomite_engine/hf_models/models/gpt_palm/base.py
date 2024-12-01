from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import GPTPaLMConfig
from .layer import GPTPaLMBlock


class GPTPaLMPreTrainedModel(PreTrainedModelMixin):
    config_class = GPTPaLMConfig
    layer_class = GPTPaLMBlock
    _no_split_modules = ["GPTPaLMBlock"]


class GPTPaLMModel(GPTPaLMPreTrainedModel, BaseModelMixin): ...
