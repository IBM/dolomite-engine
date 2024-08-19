from ...mixins import CausalLMModelMixin_TP
from .base import GPTEnsembleModel_TP, GPTEnsemblePreTrainedModel_TP


class GPTEnsembleForCausalLM_TP(GPTEnsemblePreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = GPTEnsembleModel_TP
