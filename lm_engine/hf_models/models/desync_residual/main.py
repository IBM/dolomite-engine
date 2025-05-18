from ...mixins import CausalLMModelMixin
from .base import DesyncResidualModel, DesyncResidualPreTrainedModel


class DesyncResidualForCausalLM(DesyncResidualPreTrainedModel, CausalLMModelMixin):
    base_model_class = DesyncResidualModel
