from ...mixins import CausalLMModelMixin
from .base import GPTLadderModel, GPTLadderPreTrainedModel


class GPTLadderForCausalLM(GPTLadderPreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTLadderModel
