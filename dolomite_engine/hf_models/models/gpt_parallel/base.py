from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import GPTParallelConfig
from .layer import GPTParallelBlock


class GPTParallelPreTrainedModel(PreTrainedModelMixin):
    config_class = GPTParallelConfig
    layer_class = GPTParallelBlock
    _no_split_modules = ["GPTParallelBlock"]


class GPTParallelModel(GPTParallelPreTrainedModel, BaseModelMixin): ...
