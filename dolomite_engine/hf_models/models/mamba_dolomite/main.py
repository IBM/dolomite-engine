from ...mixins import CausalLMModelMixin
from .base import Mamba2DolomiteModel, Mamba2DolomitePreTrainedModel


class Mamba2DolomiteForCausalLM(Mamba2DolomitePreTrainedModel, CausalLMModelMixin):
    base_model_class = Mamba2DolomiteModel
