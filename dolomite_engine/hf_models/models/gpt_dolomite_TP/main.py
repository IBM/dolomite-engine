from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from ....utils import SafeTensorsWeightsManager
from ...modeling_utils import ParameterizedLinear
from ...modeling_utils_TP import (
    LMHead_TP,
    TensorParallelCrossEntropy,
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
)
from ..gpt_dolomite import GPTDolomiteConfig, GPTDolomiteForCausalLM, GPTDolomitePreTrainedModel
from .base import GPTDolomiteModel_TP, GPTDolomitePreTrainedModel_TP


class GPTDolomiteForCausalLM_TP(GPTDolomitePreTrainedModel_TP, GPTDolomiteForCausalLM):
    def __init__(self, config: GPTDolomiteConfig, tensor_parallel_embeddings: bool = False, **kwargs) -> None:
        GPTDolomitePreTrainedModel.__init__(self, config, **kwargs)

        self.tensor_parallel_embeddings = tensor_parallel_embeddings
        self.vocab_size = config.vocab_size

        self.transformer = GPTDolomiteModel_TP(config, tensor_parallel_embeddings=tensor_parallel_embeddings, **kwargs)

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

    def forward(
        self,
        input_ids: torch.Tensor = None,
        past_key_values: DynamicCache = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        output_parallel_lm_logits: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert not output_attentions

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.get_lm_logits(hidden_states)

        if self.m_width is not None:
            lm_logits = lm_logits / self.m_width

        loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels)

        if output_parallel_lm_logits:
            assert self.tensor_parallel_embeddings
        else:
            if self.tensor_parallel_embeddings:
                lm_logits = gather_from_tensor_parallel_region(lm_logits)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel_embeddings:
            hidden_states = copy_to_tensor_parallel_region(hidden_states)

        lm_logits = super().get_lm_logits(hidden_states)

        return lm_logits

    def get_autoregressive_language_modeling_loss(self, lm_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels is None:
            return None

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

        if self.tensor_parallel_embeddings:
            loss = TensorParallelCrossEntropy.apply(
                shift_logits, shift_labels, self.vocab_size, self.upcast_logits_for_loss
            )
        else:
            if self.upcast_logits_for_loss:
                shift_logits = shift_logits.float()

            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        self.transformer.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "transformer.")

        if not self._tied_word_embeddings:
            if self.tensor_parallel_embeddings:
                self.lm_head.load_from_safetensors_weights_manager(safetensors_weight_manager, "lm_head.")
            else:
                state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "transformer.wte.weight")}
                self.lm_head.load_state_dict(state_dict)

    @classmethod
    def from_pretrained(
        self,
        model_name: str,
        torch_dtype: torch.dtype = torch.float32,
        tensor_parallel_embeddings: bool = False,
    ) -> GPTDolomiteForCausalLM_TP:
        # use dummy tensors to avoid initializing model here
        with torch.device("meta"):
            config = GPTDolomiteConfig.from_pretrained(model_name)
            # try sharding vocab matrices if really struggling for memory
            model = GPTDolomiteForCausalLM_TP(config, tensor_parallel_embeddings=tensor_parallel_embeddings)

            # set dtype
            model = model.to(dtype=torch_dtype)

        # copy to device without copying storage
        model = model.to_empty(device=torch.cuda.current_device())

        # load weights into tensor parallel model using SafeTensorsWeightsManager class
        # this avoids loading multiple copies of the parameters in CPU memory
        safetensors_weight_manager = SafeTensorsWeightsManager(model_name)
        model.load_from_safetensors_weights_manager(safetensors_weight_manager)

        return model
