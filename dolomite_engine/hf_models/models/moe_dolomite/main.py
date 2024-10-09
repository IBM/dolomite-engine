from ...mixins import CausalLMMoEModelMixin
from .base import MoEDolomiteModel, MoEDolomitePreTrainedModel


class MoEDolomiteForCausalLM(MoEDolomitePreTrainedModel, CausalLMMoEModelMixin):
    base_model_class = MoEDolomiteModel
