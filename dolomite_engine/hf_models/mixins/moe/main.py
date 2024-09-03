import torch
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

from ..dense import CausalLMModelMixin


class CausalLMMoEModelMixin(CausalLMModelMixin):
    def forward(
        self,
        input_ids: torch.Tensor | list[list[int]] | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | list[list[int]] | None = None,
        position_ids: torch.Tensor | list[list[int]] | None = None,
        inputs_embeds: torch.Tensor | list[list[float]] | None = None,
        labels: torch.Tensor | list[list[int]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
    ) -> tuple | MoeCausalLMOutputWithPast:
        if self._use_padding_free_transformer and output_router_logits:
            raise NotImplementedError("padding_free is not supported with load_balancing_loss_func currently")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        # ==========================================================================================
        # padding_free:
        #     input_ids -> (total_q)
        #     attention_mask -> None
        #     position_ids -> (total_q)
        # else:
        #     input_ids -> (batch_size, query_length)
        #     attention_mask -> None or (batch_size, key_length)
        #     position_ids -> None or (batch_size, key_length)
        # ==========================================================================================

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
            output_router_logits=output_router_logits,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.get_lm_logits(hidden_states)

        if self.m_width is not None:
            lm_logits = lm_logits / self.m_width

        loss, load_balancing_loss = self.get_moe_loss(
            lm_logits=lm_logits,
            labels=labels,
            cu_seqlens=cu_seqlens,
            router_logits=transformer_outputs.router_logits if return_dict else transformer_outputs[-1],
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_tok,
            router_aux_loss_coef=self.router_aux_loss_coef,
            output_router_logits=output_router_logits,
        )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            if output_router_logits:
                output = (load_balancing_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=load_balancing_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            router_logits=transformer_outputs.router_logits,
        )

    def get_moe_loss(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        cu_seqlens: torch.Tensor,
        router_logits: torch.Tensor,
        num_experts: int,
        num_experts_per_token: int,
        router_aux_loss_coef: float,
        output_router_logits: bool,
    ) -> torch.Tensor:
        loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels, cu_seqlens)

        load_balancing_loss = None
        if output_router_logits:
            load_balancing_loss = load_balancing_loss_func(router_logits, num_experts, num_experts_per_token)
            if loss is not None:
                loss += router_aux_loss_coef * load_balancing_loss

        return loss, load_balancing_loss
