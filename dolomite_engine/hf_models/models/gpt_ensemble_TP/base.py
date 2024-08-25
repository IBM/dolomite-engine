from ....utils import ProcessGroupManager
from ...mixins import PreTrainedModelMixin_TP
from ..gpt_dolomite_TP import GPTDolomiteModel_TP
from ..gpt_ensemble import GPTEnsembleConfig
from .layer import GPTEnsembleBlock_TP


class GPTEnsemblePreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTEnsembleConfig
    layer_class = GPTEnsembleBlock_TP
    _no_split_modules = ["GPTEnsembleBlock_TP"]

    def __init__(self, config: GPTEnsembleConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        assert config.pretraining_tensor_parallel_size == ProcessGroupManager.get_tensor_parallel_world_size()


class GPTEnsembleModel_TP(GPTEnsemblePreTrainedModel_TP, GPTDolomiteModel_TP): ...
