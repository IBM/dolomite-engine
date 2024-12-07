from ...mixins import CausalLMMoEModelMixin
from .base import MoELadderResidualModel, MoELadderResidualPreTrainedModel


class MoEDolomiteForCausalLM(MoELadderResidualPreTrainedModel, CausalLMMoEModelMixin):
    base_model_class = MoELadderResidualModel
