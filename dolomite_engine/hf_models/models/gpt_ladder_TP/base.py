import torch
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..gpt_ladder import GPTLadderConfig
from .layer import GPTLadderBlock_TP


class GPTLadderPreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTLadderConfig
    layer_class = GPTLadderBlock_TP
    _no_split_modules = ["GPTLadderBlock_TP"]


class GPTLadderModel_TP(GPTLadderPreTrainedModel_TP, BaseModelMixin_TP):
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
    ) -> tuple | BaseModelOutputWithPast:
        (
            output_hidden_states,
            use_cache,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
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
        )

        previous_attention_out = torch.zeros_like(hidden_states)
        previous_mlp_out = torch.zeros_like(hidden_states)

        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        all_hidden_states = () if output_hidden_states else None
        for block in self.h:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            previous_attention_out, previous_mlp_out, hidden_states = block(
                previous_attention_out=previous_attention_out,
                previous_mlp_out=previous_mlp_out,
                residual=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = hidden_states + previous_attention_out + previous_mlp_out

        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )
