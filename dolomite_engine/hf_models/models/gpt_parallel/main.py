from ...mixins import CausalLMModelMixin
from .base import GPTParallelModel, GPTParallelPreTrainedModel


class GPTParallelForCausalLM(GPTParallelPreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTParallelModel
