from typing import Tuple

from ...modeling_utils import ParameterizedLinear
from ..gpt_dolomite import GPTDolomiteForCausalLM
from .base import GPTMultiLayerModel, GPTMultiLayerPreTrainedModel
from .config import GPTMultiLayerConfig


class GPTMultiLayerForCausalLM(GPTMultiLayerPreTrainedModel, GPTDolomiteForCausalLM):
    def __init__(self, config: GPTMultiLayerConfig, **kwargs) -> None:
        GPTMultiLayerPreTrainedModel.__init__(self, config, **kwargs)

        self.transformer = GPTMultiLayerModel(config, **kwargs)

        if not self._tied_word_embeddings:
            self.lm_head = ParameterizedLinear(
                config.n_embd, config.vocab_size, bias=False, std=config.initializer_range
            )

        self.m_width = config.m_width

        # Initialize weights and apply final processing
        self.post_init()

    def get_global_local_idx(self, index: int) -> Tuple[int, int]:
        return self.transformer.layer_map[index]
