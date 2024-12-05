from ...mixins import CausalLMMoEModelMixin_TP
from ..moe_dolomite_TP.weights import get_moe_dolomite_tensor_parallel_state_dict
from .base import MoELadderResidualModel_TP, MoELadderResidualPreTrainedModel_TP


class MoELadderResidualForCausalLM_TP(MoELadderResidualPreTrainedModel_TP, CausalLMMoEModelMixin_TP):
    base_model_class = MoELadderResidualModel_TP
    model_parallel_state_dict_function = get_moe_dolomite_tensor_parallel_state_dict
