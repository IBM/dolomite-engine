from ...mixins import CausalLMMoEModelMixin
from .base import MoEStickBreakingModel, MoEStickBreakingPreTrainedModel


class MoEStickBreakingForCausalLM(MoEStickBreakingPreTrainedModel, CausalLMMoEModelMixin):
    base_model_class = MoEStickBreakingModel
