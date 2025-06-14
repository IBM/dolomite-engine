# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin
from .base import GPTBaseModel, GPTDolomitePreTrainedModel


class GPTBaseForCausalLM(GPTDolomitePreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTBaseModel
