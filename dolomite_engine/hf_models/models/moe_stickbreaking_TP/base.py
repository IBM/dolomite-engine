from ...mixins import BaseMoEModelMixin_TP, PreTrainedMoEModelMixin_TP
from ..moe_stickbreaking import MoEStickBreakingConfig
from .layer import MoEStickBreakingBlock_TP


class MoEStickBreakingPreTrainedModel_TP(PreTrainedMoEModelMixin_TP):
    config_class = MoEStickBreakingConfig
    layer_class = MoEStickBreakingBlock_TP
    _no_split_modules = ["MoEStickBreakingBlock_TP"]


class MoEStickBreakingModel_TP(MoEStickBreakingPreTrainedModel_TP, BaseMoEModelMixin_TP): ...
