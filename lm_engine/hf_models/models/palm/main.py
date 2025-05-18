from ...mixins import CausalLMModelMixin
from .base import PaLMModel, PaLMPreTrainedModel


class PaLMForCausalLM(PaLMPreTrainedModel, CausalLMModelMixin):
    base_model_class = PaLMModel
