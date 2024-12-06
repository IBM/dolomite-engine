import torch
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...mixins import BaseMoEModelMixin, PreTrainedModelMixin
from ..stickbreaking import StickBreakingPreTrainedModel
from .config import MoEStickBreakingConfig
from .layer import MoEStickBreakingBlock


class MoEStickBreakingPreTrainedModel(PreTrainedModelMixin):
    config_class = MoEStickBreakingConfig
    layer_class = MoEStickBreakingBlock
    _no_split_modules = ["MoEStickBreakingBlock"]


class MoEStickBreakingModel(StickBreakingPreTrainedModel, BaseMoEModelMixin): ...
