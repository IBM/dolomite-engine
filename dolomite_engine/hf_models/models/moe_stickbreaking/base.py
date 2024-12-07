import torch
from transformers import DynamicCache

from ...mixins import BaseMoEModelMixin, MoeModelOutputWithPastAndAuxLoss, PreTrainedMoEModelMixin
from ..stickbreaking.sb_varlen import BLOCK_M, BLOCK_N, row_block_counts_and_sequence_ids
from .config import MoEStickBreakingConfig
from .layer import MoEStickBreakingBlock


class MoEStickBreakingPreTrainedModel(PreTrainedMoEModelMixin):
    config_class = MoEStickBreakingConfig
    layer_class = MoEStickBreakingBlock
    _no_split_modules = ["MoEStickBreakingBlock"]


class MoEStickBreakingModel(MoEStickBreakingPreTrainedModel, BaseMoEModelMixin):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
        output_aux_loss: bool = True,
    ) -> MoeModelOutputWithPastAndAuxLoss:
        (
            output_hidden_states,
            use_cache,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
            output_router_logits,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
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

        sb_metadata = None
        if self._use_padding_free_transformer:
            with torch.no_grad():
                # cu_row_blocks, first_row_block, sequence_ids
                sb_metadata = row_block_counts_and_sequence_ids(cu_seqlens[1:], BLOCK_M, BLOCK_N)

        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None
        total_aux_loss = 0

        for block in self.h:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            outputs = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_router_logits=output_router_logits,
                output_aux_loss=output_aux_loss,
                sb_metadata=sb_metadata,
            )

            hidden_states = outputs[0]
            outputs = outputs[1:]

            if output_router_logits:
                all_router_logits += (outputs[0],)
                outputs = outputs[1:]

            if output_aux_loss:
                aux_loss = outputs[0]
                total_aux_loss = total_aux_loss + aux_loss

        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return MoeModelOutputWithPastAndAuxLoss(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            router_logits=all_router_logits,
            aux_loss=total_aux_loss,
        )
