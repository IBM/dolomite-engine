from ...mixins import CausalLMModelMixin_TP
from .base import GPTParallelModel_TP, GPTParallelPreTrainedModel_TP


class GPTParallelForCausalLM_TP(GPTParallelPreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = GPTParallelModel_TP
