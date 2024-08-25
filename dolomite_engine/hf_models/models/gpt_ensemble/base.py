import torch
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...enums import PositionEmbeddingType
from ...mixins import PreTrainedModelMixin
from ..gpt_dolomite import GPTDolomiteModel
from .config import GPTEnsembleConfig
from .layer import GPTEnsembleBlock


class GPTEnsemblePreTrainedModel(PreTrainedModelMixin):
    config_class = GPTEnsembleConfig
    layer_class = GPTEnsembleBlock
    _no_split_modules = ["GPTEnsembleBlock"]


class GPTEnsembleModel(GPTEnsemblePreTrainedModel, GPTDolomiteModel):
    def __init__(self, config: GPTEnsembleConfig, **kwargs) -> None:
        self.tensor_parallel_size = config.pretraining_tensor_parallel_size

        super().__init__(config, **kwargs)

        if self.position_embedding_type == PositionEmbeddingType.alibi:
            raise NotImplementedError("currently GPTEnsemble doesn't support alibi")

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
        return_dict: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        (
            output_hidden_states,
            use_cache,
            return_dict,
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
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = hidden_states.unsqueeze(0)

        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
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

        hidden_states = hidden_states.squeeze(0)
        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    def _get_rope_cos_sin(
        self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        rope_cos_sin = super()._get_rope_cos_sin(key_length, position_ids, dtype=dtype, device=device)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            cos, sin = rope_cos_sin

            if position_ids.shape[0] != 1:
                cos = cos.repeat(self.tensor_parallel_size, 1, 1, 1)
                sin = sin.repeat(self.tensor_parallel_size, 1, 1, 1)

            return cos, sin
