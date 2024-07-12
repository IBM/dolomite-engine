from ...modeling_utils import ParameterizedLinear
from ..gpt_dolomite import GPTDolomiteForCausalLM
from .base import RNNDolomiteModel, RNNDolomitePreTrainedModel
from .config import RNNDolomiteConfig


class RNNDolomiteForCausalLM(RNNDolomitePreTrainedModel, GPTDolomiteForCausalLM):
    def __init__(self, config: RNNDolomiteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.transformer = RNNDolomiteModel(config, **kwargs)

        if not self._tied_word_embeddings:
            self.lm_head = ParameterizedLinear(
                config.n_embd, config.vocab_size, bias=False, std=config.initializer_range
            )

        self.m_width = config.m_width
        self.upcast_logits_for_loss = config.upcast_logits_for_loss

        # Initialize weights and apply final processing
        self.post_init()
