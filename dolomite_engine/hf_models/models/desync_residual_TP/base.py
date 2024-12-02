from ....utils import ProcessGroupManager
from ...mixins import PreTrainedModelMixin_TP
from ..desync_residual import DesyncResidualConfig
from ..gpt_dolomite_TP import GPTDolomiteModel_TP
from .layer import DesyncResidualBlock_TP


class DesyncResidualPreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = DesyncResidualConfig
    layer_class = DesyncResidualBlock_TP
    _no_split_modules = ["DesyncResidualBlock_TP"]

    def __init__(self, config: DesyncResidualConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        assert config.pretraining_tensor_parallel_size == ProcessGroupManager.get_tensor_parallel_world_size()


class DesyncResidualModel_TP(DesyncResidualPreTrainedModel_TP, GPTDolomiteModel_TP): ...
