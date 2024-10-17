from ...mixins import CausalLMMoEModelMixin
from .base import RNNMoEDolomiteModel, RNNMoEDolomitePreTrainedModel


class RNNMoEDolomiteForCausalLM(RNNMoEDolomitePreTrainedModel, CausalLMMoEModelMixin):
    base_model_class = RNNMoEDolomiteModel
