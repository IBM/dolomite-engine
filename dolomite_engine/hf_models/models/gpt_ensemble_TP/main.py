from ...mixins import CausalLMModelMixin_TP
from .base import GPTEnsembleModel_TP, GPTEnsemblePreTrainedModel_TP
from .weights import get_gpt_ensemble_model_parallel_state_dict


class GPTEnsembleForCausalLM_TP(GPTEnsemblePreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = GPTEnsembleModel_TP
    model_parallel_state_dict_function = get_gpt_ensemble_model_parallel_state_dict
