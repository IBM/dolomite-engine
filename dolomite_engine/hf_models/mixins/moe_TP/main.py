import torch
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from transformers import DynamicCache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from ....distributed import dtensor_to_tensor, tensor_to_dtensor
from ...loss import get_autoregressive_language_modeling_loss
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
        reduction: str = "mean",
        output_router_logits: bool | None = None,
        prev_aux_loss: torch.Tensor | None = None,
    ) -> tuple | MoeCausalLMOutputWithPast:
        # Note: `past_key_values` contains aux_loss in non-stage-one PP

        if not self.is_pipeline_parallel_enabled or self.is_first_stage:
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

        if not self.is_pipeline_parallel_enabled or self.is_last_stage:
            lm_logits = self.get_lm_logits(transformer_outputs.last_hidden_state)

            if self.m_width is not None:
                lm_logits = lm_logits / self.m_width

        if not self.is_pipeline_parallel_enabled:
            lm_loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels, cu_seqlens)

        aux_loss = transformer_outputs.aux_loss
        # aux_loss = tensor_to_dtensor(transformer_outputs.aux_loss, device_mesh=self.tp_mesh, current_placement=Replicate())
        # aux_loss = dtensor_to_tensor(aux_loss, device_mesh=self.tp_mesh, desired_placement=Replicate())
        if self.is_pipeline_parallel_enabled and not self.is_first_stage:
            # Non-first-stage PP will have aux loss as `past_key_values` due to the way pipeline works
            # So accumulate aux_loss
            aux_loss = aux_loss + prev_aux_loss

        if not self.is_pipeline_parallel_enabled or self.is_last_stage:
            if output_parallel_lm_logits:
                assert self.tensor_parallel_word_embeddings
            else:
                if self.tensor_parallel_word_embeddings:
                    # all gather
                    lm_logits = tensor_to_dtensor(lm_logits, device_mesh=self.tp_mesh, current_placement=Shard(-1))
                    lm_logits = dtensor_to_tensor(lm_logits, device_mesh=self.tp_mesh, desired_placement=Replicate())

        if not self.is_pipeline_parallel_enabled:
            if lm_loss is None:
                loss = None
            else:
                loss = lm_loss + self.router_aux_loss_coef * aux_loss
            output = MoeCausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                router_logits=transformer_outputs.router_logits,
            )
        elif self.is_last_stage:
            output = (lm_logits, aux_loss)
        else:
            output = (transformer_outputs.last_hidden_state, aux_loss)

        return output

    def get_dummy_input_tensor(
        self, micro_batch_size: int, sequence_length: int, intermediate_dtype: torch.dtype
    ) -> tuple[torch.Tensor] | torch.Tensor:

        if self.is_first_stage:
            # 1 is added to sequence length since megatron's dataloader gives an extra token and for good reason
            dummy_input = torch.empty(
                micro_batch_size, sequence_length + 1, device=torch.cuda.current_device(), dtype=torch.long
            )
            return dummy_input
        else:
            dummy_input = super().get_dummy_input_tensor(micro_batch_size, sequence_length, intermediate_dtype)
            aux_loss_dummy = torch.tensor(0.0, device=torch.cuda.current_device(), dtype=intermediate_dtype)
            if isinstance(dummy_input, tuple):
                return dummy_input + (aux_loss_dummy,)
            else:
                return (dummy_input, aux_loss_dummy)

    def get_dummy_output_tensor(
        self,
        micro_batch_size: int,
        sequence_length: int,
        intermediate_dtype: torch.dtype,
        output_parallel_lm_logits_if_possible: bool,
    ) -> tuple[int]:
        dummy_output = super().get_dummy_output_tensor(
            micro_batch_size, sequence_length, intermediate_dtype, output_parallel_lm_logits_if_possible
        )
        dummy_aux_loss = torch.tensor(0.0, device=torch.cuda.current_device(), dtype=intermediate_dtype)
        if isinstance(dummy_output, tuple):
            dummy_output = dummy_output + (dummy_aux_loss,)
        else:
            dummy_output = (dummy_output, dummy_aux_loss)
        return dummy_output
