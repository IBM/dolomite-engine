from __future__ import annotations

from ...modeling_utils import ParameterizedLinear
from ...modeling_utils_TP import LMHead_TP
from ..gpt_dolomite import GPTDolomiteConfig
from ..gpt_dolomite_TP import GPTDolomiteForCausalLM_TP
from .base import GPTEnsembleModel_TP, GPTEnsemblePreTrainedModel_TP


class GPTEnsembleForCausalLM_TP(GPTEnsemblePreTrainedModel_TP, GPTDolomiteForCausalLM_TP):
    def __init__(self, config: GPTDolomiteConfig, tensor_parallel_embeddings: bool = False, **kwargs) -> None:
        GPTEnsemblePreTrainedModel_TP.__init__(self, config, **kwargs)

        self.tensor_parallel_embeddings = tensor_parallel_embeddings
        self.vocab_size = config.vocab_size

        self.transformer = GPTEnsembleModel_TP(config, tensor_parallel_embeddings=tensor_parallel_embeddings, **kwargs)

        if not self._tied_word_embeddings:
            if self.tensor_parallel_embeddings:
                self.lm_head = LMHead_TP(config.vocab_size, config.n_embd, std=config.initializer_range)
            else:
                self.lm_head = ParameterizedLinear(
                    config.n_embd, config.vocab_size, bias=False, std=config.initializer_range
                )

        self.m_width = config.m_width
        self.upcast_logits_for_loss = config.upcast_logits_for_loss

        # Initialize weights and apply final processing
        self.post_init()
