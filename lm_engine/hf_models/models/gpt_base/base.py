# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import GPTBaseConfig


class GPTDolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = GPTBaseConfig


class GPTBaseModel(GPTDolomitePreTrainedModel, BaseModelMixin): ...
