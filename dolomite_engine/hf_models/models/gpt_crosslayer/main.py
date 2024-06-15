from typing import Tuple

from ...modeling_utils import ParameterizedLinear
from ..gpt_dolomite import GPTDolomiteForCausalLM
from .base import GPTCrossLayerModel, GPTCrossLayerPreTrainedModel
from .config import GPTCrossLayerConfig


class GPTCrossLayerForCausalLM(GPTCrossLayerPreTrainedModel, GPTDolomiteForCausalLM):
    def __init__(self, config: GPTCrossLayerConfig, **kwargs) -> None:
        GPTCrossLayerPreTrainedModel.__init__(self, config, **kwargs)

        self.transformer = GPTCrossLayerModel(config, **kwargs)

        if not self._tied_word_embeddings:
            self.lm_head = ParameterizedLinear(
                config.n_embd, config.vocab_size, bias=False, std=config.initializer_range
            )

        self.m_width = config.m_width
        self.upcast_logits_for_loss = config.upcast_logits_for_loss

        # Initialize weights and apply final processing
        self.post_init()

    def get_global_local_idx(self, index: int) -> Tuple[int, int]:
        return self.transformer.layer_map[index]
