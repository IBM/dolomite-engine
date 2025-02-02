import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .cache import HybridMambaAttentionDynamicCache
from .config import Mamba2DolomiteConfig
from .layer import Mamba2DolomiteBlock


class Mamba2DolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = Mamba2DolomiteConfig
    layer_class = Mamba2DolomiteBlock
    _no_split_modules = ["Mamba2DolomiteBlock"]


class Mamba2DolomiteModel(Mamba2DolomitePreTrainedModel, BaseModelMixin):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: HybridMambaAttentionDynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
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

        # ==========================================================================================
        # padding_free:
        #     attention_mask -> None
        # flash:
        #     attention_mask -> (batch_size, key_length)
        # else:
        #     attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        past_key_values = (
            HybridMambaAttentionDynamicCache(config=self.config, batch_size=input_ids.size(0))
            if use_cache and past_key_values is None
            else past_key_values
        )
        all_hidden_states = () if output_hidden_states else None
        for block in self.h:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    def _update_mamba_mask(
        self, attention_mask: torch.Tensor | None, past_key_values: HybridMambaAttentionDynamicCache
    ) -> torch.Tensor | None:
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None
        return mamba_mask
