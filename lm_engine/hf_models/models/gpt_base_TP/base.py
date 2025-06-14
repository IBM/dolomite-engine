# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..gpt_base import GPTBaseConfig


class GPTDolomitePreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTBaseConfig


class GPTBaseModel_TP(GPTDolomitePreTrainedModel_TP, BaseModelMixin_TP): ...
