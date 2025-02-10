from ...mixins import CausalLMModelMixin
from .base import GPTCrossLayerModel, GPTCrossLayerPreTrainedModel


class GPTCrossLayerForCausalLM(GPTCrossLayerPreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTCrossLayerModel
