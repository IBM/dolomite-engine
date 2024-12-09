from ...mixins import CausalLMModelMixin
from .base import StickBreakingModel, StickBreakingPreTrainedModel


class StickBreakingForCausalLM(StickBreakingPreTrainedModel, CausalLMModelMixin):
    base_model_class = StickBreakingModel
