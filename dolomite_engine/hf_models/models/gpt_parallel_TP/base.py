from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..gpt_parallel import GPTParallelConfig
from .layer import GPTParallelBlock_TP


class GPTParallelPreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTParallelConfig
    layer_class = GPTParallelBlock_TP
    _no_split_modules = ["GPTParallelBlock_TP"]


class GPTParallelModel_TP(GPTParallelPreTrainedModel_TP, BaseModelMixin_TP): ...
