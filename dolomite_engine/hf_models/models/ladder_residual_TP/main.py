from ...mixins import CausalLMModelMixin_TP
from .base import LadderResidualModel_TP, LadderResidualPreTrainedModel_TP


class LadderResidualForCausalLM_TP(LadderResidualPreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = LadderResidualModel_TP
