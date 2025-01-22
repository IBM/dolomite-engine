from ...mixins import CausalLMModelMixin
from .base import MambaDolomiteModel, MambaDolomitePreTrainedModel


class MambaDolomiteForCausalLM(MambaDolomitePreTrainedModel, CausalLMModelMixin):
    base_model_class = MambaDolomiteModel
