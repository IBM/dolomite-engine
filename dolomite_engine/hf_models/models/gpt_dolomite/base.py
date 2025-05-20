# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import GPTDolomiteConfig


class GPTDolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = GPTDolomiteConfig


class GPTDolomiteModel(GPTDolomitePreTrainedModel, BaseModelMixin): ...
