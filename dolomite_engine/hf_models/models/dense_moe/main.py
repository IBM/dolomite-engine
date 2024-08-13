import torch
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...modeling_utils import ParameterizedLinear
from ..gpt_dolomite import GPTDolomiteForCausalLM
from .base import DenseMoEModel, DenseMoEPreTrainedModel
from .config import DenseMoEConfig


class DenseMoEForCausalLM(DenseMoEPreTrainedModel, GPTDolomiteForCausalLM):
    def __init__(self, config: DenseMoEConfig, **kwargs) -> None:
        DenseMoEPreTrainedModel.__init__(self, config, **kwargs)

        self.transformer = DenseMoEModel(config, **kwargs)

        if not self._tied_word_embeddings:
            self.lm_head = ParameterizedLinear(
                config.n_embd, config.vocab_size, bias=False, std=config.initializer_range
            )

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts

        self.m_width = config.m_width
        self.upcast_logits_for_loss = config.upcast_logits_for_loss

        # Initialize weights and apply final processing
        self.post_init()
