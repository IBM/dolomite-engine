from ...mixins import CausalLMModelMixin_TP
from .base import GPTLadderModel_TP, GPTLadderPreTrainedModel_TP


class GPTLadderForCausalLM_TP(GPTLadderPreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = GPTLadderModel_TP
