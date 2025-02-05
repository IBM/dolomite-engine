from ...mixins import CausalLMModelMixin
from .base import RNNMoEDolomiteModel, RNNMoEDolomitePreTrainedModel


class RNNMoEDolomiteForCausalLM(RNNMoEDolomitePreTrainedModel, CausalLMModelMixin):
    base_model_class = RNNMoEDolomiteModel
