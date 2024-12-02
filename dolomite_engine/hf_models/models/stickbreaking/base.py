from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import StickBreakingConfig
from .layer import StickBreakingBlock


class StickBreakingPreTrainedModel(PreTrainedModelMixin):
    config_class = StickBreakingConfig
    layer_class = StickBreakingBlock
    _no_split_modules = ["StickBreakingBlock"]


class StickBreakingModel(StickBreakingPreTrainedModel, BaseModelMixin): ...
