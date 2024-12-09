from ...mixins import CausalLMMoEModelMixin
from .base import MoELadderResidualModel, MoELadderResidualPreTrainedModel


class MoELadderResidualForCausalLM(MoELadderResidualPreTrainedModel, CausalLMMoEModelMixin):
    base_model_class = MoELadderResidualModel
