from ...mixins import CausalLMModelMixin
from .base import GPTPaLMModel, GPTPaLMPreTrainedModel


class GPTPaLMForCausalLM(GPTPaLMPreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTPaLMModel
