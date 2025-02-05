from ...mixins import CausalLMModelMixin
from .base import MoEStickBreakingModel, MoEStickBreakingPreTrainedModel


class MoEStickBreakingForCausalLM(MoEStickBreakingPreTrainedModel, CausalLMModelMixin):
    base_model_class = MoEStickBreakingModel
