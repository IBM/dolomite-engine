from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from ....utils import ProcessGroupManager, SafeTensorsWeightsManager
from ...modeling_utils_TP import LMHead_TP, dtensor_to_tensor, tensor_to_dtensor
from ..gpt_dolomite import GPTDolomiteConfig, GPTDolomiteForCausalLM
from .base import GPTDolomiteModel_TP, GPTDolomitePreTrainedModel_TP


class GPTDolomiteForCausalLM_TP(GPTDolomitePreTrainedModel_TP, GPTDolomiteForCausalLM):
    def __init__(self, config: GPTDolomiteConfig, **kwargs) -> None:
        GPTDolomitePreTrainedModel_TP.__init__(self, config, **kwargs)

        self.vocab_size = config.vocab_size

        self.transformer = GPTDolomiteModel_TP(config, **kwargs)

        if not self._tied_word_embeddings:
            self.lm_head = LMHead_TP(
                self.vocab_size,
                config.n_embd,
                std=config.initializer_range,
                tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
                sequence_parallel=self.sequence_parallel,
            )

        self.m_width = config.m_width
        self.upcast_logits_for_loss = config.upcast_logits_for_loss

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | list[list[int]] | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | list[list[int]] | None = None,
        position_ids: torch.Tensor | list[list[int]] | None = None,
        inputs_embeds: torch.Tensor | list[list[float]] | None = None,
        labels: torch.Tensor | list[list[int]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        output_parallel_lm_logits: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert not output_attentions

        input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen = self.prepare_inputs_for_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

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
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.get_lm_logits(hidden_states)

        if self.m_width is not None:
            lm_logits = lm_logits / self.m_width

        loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels, cu_seqlens)

        if output_parallel_lm_logits:
            assert self.tensor_parallel_word_embeddings
        else:
            if self.tensor_parallel_word_embeddings:
                # all gather
                lm_logits = tensor_to_dtensor(lm_logits, current_placement=Shard(-1))
                lm_logits = dtensor_to_tensor(lm_logits, desired_placement=Replicate())

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
        return (
            LMHead_TP.compute_with_weight(
                hidden_states,
                weight=self.transformer.wte.weight,
                tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
                use_padding_free_transformer=self._use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )
            if self._tied_word_embeddings
            else self.lm_head(hidden_states)
        )

    def get_autoregressive_language_modeling_loss(
        self, lm_logits: torch.Tensor, labels: torch.Tensor | None, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        if labels is None:
            return None

        if self._use_padding_free_transformer:
            shift_logits = lm_logits[:-1, :]
            shift_labels = labels[1:].to(shift_logits.device)

            # this is needed so that the last token of current example doesn't predict first token of next example
            drop_loss_positions = cu_seqlens[1:-1] - 1
            shift_labels[drop_loss_positions] = -100
        else:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

        shift_logits = tensor_to_dtensor(
            shift_logits, current_placement=Shard(-1) if self.tensor_parallel_word_embeddings else Replicate()
        )
        shift_labels = tensor_to_dtensor(shift_labels, current_placement=Replicate())

        if self.upcast_logits_for_loss:
            shift_logits = shift_logits.float()

        loss_context = loss_parallel if self.tensor_parallel_word_embeddings else nullcontext
        with loss_context():
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        self.transformer.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "transformer.")

        if not self._tied_word_embeddings:
            if self.tensor_parallel_word_embeddings:
                self.lm_head.load_from_safetensors_weights_manager(safetensors_weight_manager, "lm_head.")
            else:
                state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "transformer.wte.weight")}
                self.lm_head.load_state_dict(state_dict)

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float32,
        tensor_parallel_word_embeddings: bool = False,
        **kwargs,
    ) -> GPTDolomiteForCausalLM_TP:
        config = GPTDolomiteConfig.from_pretrained(pretrained_model_name_or_path)

        # use dummy tensors to avoid initializing model here
        with torch.device("meta"):
            # try sharding vocab matrices if really struggling for memory
            model = GPTDolomiteForCausalLM_TP._from_config(
                config, tensor_parallel_word_embeddings=tensor_parallel_word_embeddings, **kwargs
            )
            model = model.to(dtype=torch_dtype)

        # copy to device without copying storage
        model = model.to_empty(device=torch.cuda.current_device())

        # load weights into tensor parallel model using SafeTensorsWeightsManager class
        # this avoids loading multiple copies of the parameters in CPU memory
        safetensors_weight_manager = SafeTensorsWeightsManager(pretrained_model_name_or_path)
        model.load_from_safetensors_weights_manager(safetensors_weight_manager)

        return model
