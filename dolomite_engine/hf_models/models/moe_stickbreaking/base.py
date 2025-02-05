import torch
from transformers import DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast

from ...loss import add_aux_loss, clear_aux_loss
from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import MoEStickBreakingConfig
from .layer import MoEStickBreakingBlock


class MoEStickBreakingPreTrainedModel(PreTrainedModelMixin):
    config_class = MoEStickBreakingConfig
    layer_class = MoEStickBreakingBlock
    _no_split_modules = ["MoEStickBreakingBlock"]


class MoEStickBreakingModel(MoEStickBreakingPreTrainedModel, BaseModelMixin):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> MoeModelOutputWithPast:
        (
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
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        sb_metadata = None
        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        clear_aux_loss()

        for block in self.h:
            hidden_states, aux_loss = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                sb_metadata=sb_metadata,
            )

            add_aux_loss(aux_loss)

        hidden_states = self.ln_f(hidden_states)

        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
