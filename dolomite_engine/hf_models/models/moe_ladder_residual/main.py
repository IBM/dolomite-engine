from ...mixins import CausalLMModelMixin
from .base import MoELadderResidualModel, MoELadderResidualPreTrainedModel


class MoELadderResidualForCausalLM(MoELadderResidualPreTrainedModel, CausalLMModelMixin):
    base_model_class = MoELadderResidualModel
