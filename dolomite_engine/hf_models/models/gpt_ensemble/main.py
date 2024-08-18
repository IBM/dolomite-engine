from ...mixins import CausalLMModelMixin
from .base import GPTEnsembleModel, GPTEnsemblePreTrainedModel


class GPTEnsembleForCausalLM(GPTEnsemblePreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTEnsembleModel
