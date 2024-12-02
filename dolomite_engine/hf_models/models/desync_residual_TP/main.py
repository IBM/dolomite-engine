from ...mixins import CausalLMModelMixin_TP
from .base import DesyncResidualModel_TP, DesyncResidualPreTrainedModel_TP
from .weights import get_desync_residual_model_parallel_state_dict


class DesyncResidualForCausalLM_TP(DesyncResidualPreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = DesyncResidualModel_TP
    model_parallel_state_dict_function = get_desync_residual_model_parallel_state_dict
