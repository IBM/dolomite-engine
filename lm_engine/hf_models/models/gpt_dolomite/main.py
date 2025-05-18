from ...mixins import CausalLMModelMixin
from .base import GPTDolomiteModel, GPTDolomitePreTrainedModel


class GPTDolomiteForCausalLM(GPTDolomitePreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTDolomiteModel
