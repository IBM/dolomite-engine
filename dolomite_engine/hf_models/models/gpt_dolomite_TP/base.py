from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..gpt_dolomite import GPTDolomiteConfig
from .layer import GPTDolomiteBlock_TP


class GPTDolomitePreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTDolomiteConfig
    layer_class = GPTDolomiteBlock_TP
    _no_split_modules = ["GPTDolomiteBlock_TP"]

    def __init__(self, config: GPTDolomiteConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.tensor_parallel_word_embeddings = kwargs.get("tensor_parallel_word_embeddings", False)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)


class GPTDolomiteModel_TP(GPTDolomitePreTrainedModel_TP, BaseModelMixin_TP): ...
