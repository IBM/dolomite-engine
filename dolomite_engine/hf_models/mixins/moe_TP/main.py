import torch
from torch.distributed._tensor.placement_types import Replicate, Shard
from transformers import DynamicCache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from ....distributed import dtensor_to_tensor, tensor_to_dtensor
from ..dense_TP import CausalLMModelMixin_TP
from ..moe import CausalLMMoEModelMixin, MoeModelOutputWithPastAndAuxLoss


class CausalLMMoEModelMixin_TP(CausalLMMoEModelMixin, CausalLMModelMixin_TP):
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
        return_dict: bool = True,
        output_parallel_lm_logits: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
    ) -> tuple | MoeCausalLMOutputWithPast:
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

        transformer_outputs: MoeModelOutputWithPastAndAuxLoss = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_router_logits=output_router_logits,
        )

        lm_logits = self.get_lm_logits(transformer_outputs.last_hidden_state)

        if self.m_width is not None:
            lm_logits = lm_logits / self.m_width

        lm_loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels, cu_seqlens)
        aux_loss = tensor_to_dtensor(
            transformer_outputs.aux_loss, device_mesh=self.tp_mesh, current_placement=Replicate()
        )

        if lm_loss is None:
            loss = None
        else:
            loss = lm_loss + self.router_aux_loss_coef * aux_loss

        if output_parallel_lm_logits:
            assert self.tensor_parallel_word_embeddings
        else:
            if self.tensor_parallel_word_embeddings:
                # all gather
                lm_logits = tensor_to_dtensor(lm_logits, device_mesh=self.tp_mesh, current_placement=Shard(-1))
                lm_logits = dtensor_to_tensor(lm_logits, device_mesh=self.tp_mesh, desired_placement=Replicate())

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            router_logits=transformer_outputs.router_logits,
        )
