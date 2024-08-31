from ...mixins import CausalLMModelMixin_TP
from .base import GPTDolomiteModel_TP, GPTDolomitePreTrainedModel_TP


class GPTDolomiteForCausalLM_TP(GPTDolomitePreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = GPTDolomiteModel_TP
