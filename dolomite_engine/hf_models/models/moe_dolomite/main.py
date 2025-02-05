from ...mixins import CausalLMModelMixin
from .base import MoEDolomiteModel, MoEDolomitePreTrainedModel


class MoEDolomiteForCausalLM(MoEDolomitePreTrainedModel, CausalLMModelMixin):
    base_model_class = MoEDolomiteModel
