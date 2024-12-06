import torch
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...mixins import PreTrainedModelMixin, BaseMoEModelMixin
from .config import MoEStickBreakingConfig
from .layer import MoEStickBreakingBlock

from ..stickbreaking import StickBreakingPreTrainedModel


class MoEStickBreakingPreTrainedModel(PreTrainedModelMixin):
    config_class = MoEStickBreakingConfig
    layer_class = MoEStickBreakingBlock
    _no_split_modules = ["MoEStickBreakingBlock"]


class MoEStickBreakingModel(StickBreakingPreTrainedModel, BaseMoEModelMixin):...
