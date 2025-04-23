from ...mixins import CausalLMModelMixin_TP
from .base import GPTDolomiteModel_TP, GPTDolomitePreTrainedModel_TP
from .weights import get_gpt_dolomite_model_parallel_state_dict


class GPTDolomiteForCausalLM_TP(GPTDolomitePreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = GPTDolomiteModel_TP
    model_parallel_state_dict_function = get_gpt_dolomite_model_parallel_state_dict
