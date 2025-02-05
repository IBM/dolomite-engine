import torch
import torch.nn.functional as F
from transformers import DynamicCache, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from ...config import CommonConfig
from ...loss import get_autoregressive_language_modeling_loss, get_aux_loss
from ...modeling_utils import ParameterizedEmbedding, ParameterizedLinear
from .base import PreTrainedModelMixin


class CausalLMModelMixin(PreTrainedModelMixin, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    base_model_class = None

    def __init__(self, config: CommonConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._init_model(config, **kwargs)

    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.transformer = self.base_model_class(config, **kwargs)

        if not self._tied_word_embeddings:
            self.lm_head = ParameterizedLinear(
                config.n_embd, config.vocab_size, bias=False, std=config.initializer_range
            )

        self.m_width = config.m_width
        self.upcast_logits_for_loss = config.upcast_logits_for_loss

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ParameterizedEmbedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: ParameterizedEmbedding) -> None:
        self.transformer.wte = value

    def get_output_embeddings(self) -> ParameterizedLinear:
        if not self._tied_word_embeddings:
            return self.lm_head

    def set_output_embeddings(self, new_embeddings: ParameterizedLinear) -> None:
        if not self._tied_word_embeddings:
            self.lm_head = new_embeddings

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
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> CausalLMOutputWithPast:
        assert return_dict

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

        transformer_outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        lm_logits = self.get_lm_logits(transformer_outputs.last_hidden_state)

        if self.m_width is not None:
            lm_logits = lm_logits / self.m_width

        loss = None
        if labels is not None:
            loss = get_autoregressive_language_modeling_loss(
                lm_logits=lm_logits,
                labels=labels,
                upcast_logits_for_loss=self.upcast_logits_for_loss,
                cu_seqlens=cu_seqlens,
                use_padding_free_transformer=self._use_padding_free_transformer,
                reduction=reduction,
            )

        if loss is None:
            loss = None
        else:
            loss = loss + self.router_aux_loss_coef * get_aux_loss()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (
            F.linear(hidden_states, self.transformer.wte.weight)
            if self._tied_word_embeddings
            else self.lm_head(hidden_states)
        )
