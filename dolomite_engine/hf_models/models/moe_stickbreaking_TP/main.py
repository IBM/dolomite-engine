from ...mixins import CausalLMMoEModelMixin_TP
from .base import MoEStickBreakingModel_TP, MoEStickBreakingPreTrainedModel_TP
from .weights import get_moe_dolomite_tensor_parallel_state_dict


class MoEStickBreakingForCausalLM_TP(MoEStickBreakingPreTrainedModel_TP, CausalLMMoEModelMixin_TP):
    base_model_class = MoEStickBreakingModel_TP
    model_parallel_state_dict_function = get_moe_dolomite_tensor_parallel_state_dict
