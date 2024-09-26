from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import GPTDolomiteConfig
from .layer import GPTDolomiteBlock


class GPTDolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = GPTDolomiteConfig
    layer_class = GPTDolomiteBlock
    _no_split_modules = ["GPTDolomiteBlock"]


class GPTDolomiteModel(GPTDolomitePreTrainedModel, BaseModelMixin): ...
