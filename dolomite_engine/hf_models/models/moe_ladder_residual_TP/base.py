from ...mixins import BaseMoEModelMixin_TP, PreTrainedMoEModelMixin_TP
from ..moe_ladder_residual import MoELadderResidualConfig
from .layer import MoELadderResidualBlock_TP


class MoELadderResidualPreTrainedModel_TP(PreTrainedMoEModelMixin_TP):
    config_class = MoELadderResidualConfig
    layer_class = MoELadderResidualBlock_TP
    _no_split_modules = ["MoELadderResidualBlock_TP"]


class MoELadderResidualModel_TP(MoELadderResidualPreTrainedModel_TP, BaseMoEModelMixin_TP): ...
