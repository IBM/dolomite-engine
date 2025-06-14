# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin_TP
from .base import GPTBaseModel_TP, GPTDolomitePreTrainedModel_TP
from .weights import get_gpt_dolomite_model_parallel_state_dict


class GPTBaseForCausalLM_TP(GPTDolomitePreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = GPTBaseModel_TP
    model_parallel_state_dict_function = get_gpt_dolomite_model_parallel_state_dict
