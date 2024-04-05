from typing import Tuple

import torch.nn as nn

from ..gpt_megatron import GPTMegatronForCausalLM
from .base import GPTMultiLayerModel, GPTMultiLayerPreTrainedModel
from .config import GPTMultiLayerConfig


class GPTMultiLayerForCausalLM(GPTMultiLayerPreTrainedModel, GPTMegatronForCausalLM):
    def __init__(self, config: GPTMultiLayerConfig, **kwargs) -> None:
        GPTMultiLayerPreTrainedModel.__init__(self, config, **kwargs)

        self.transformer = GPTMultiLayerModel(config, **kwargs)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_global_local_idx(self, index: int) -> Tuple[int, int]:
        return self.transformer.layer_map[index]
