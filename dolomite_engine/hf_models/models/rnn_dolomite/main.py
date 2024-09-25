from ...mixins import CausalLMModelMixin
from .base import RNNDolomiteModel, RNNDolomitePreTrainedModel


class RNNDolomiteForCausalLM(RNNDolomitePreTrainedModel, CausalLMModelMixin):
    base_model_class = RNNDolomiteModel
