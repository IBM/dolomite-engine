from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast

from ....utils import divide_if_divisible
from ...config import CommonConfig
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function
from ..dense import BaseModelMixin, PreTrainedModelMixin


@dataclass
class MoeModelOutputWithPastAndAuxLoss(MoeModelOutputWithPast):
    aux_loss: torch.Tensor | None = None


class PreTrainedMoEModelMixin(PreTrainedModelMixin):
    def __init__(self, config: CommonConfig, *args, **kwargs) -> None:
        self.moe_implementation = kwargs.get("moe_implementation", "eager")
        assert self.moe_implementation in ["eager", "scattermoe"]

        super().__init__(config, *args, **kwargs)


class BaseMoEModelMixin(BaseModelMixin):
    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range

        self.head_dim = divide_if_divisible(
            self.embed_dim,
            self.num_heads,
            f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                self.layer_class(
                    config,
                    attention_implementation=self.attention_implementation,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    moe_implementation=self.moe_implementation,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )
        self.ln_f = get_normalization_function(
            config.normalization_function, self.embed_dim, eps=config.layer_norm_epsilon
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

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
        output_router_logits: bool | None = None,
        output_aux_loss: bool = True,
    ) -> MoeModelOutputWithPastAndAuxLoss:
        (
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
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_router_logits=output_router_logits,
        )

        # ==========================================================================================
        # padding_free:
        #     attention_mask -> None
        # flash:
        #     attention_mask -> (batch_size, key_length)
        # else:
        #     attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        all_router_logits = () if output_router_logits else None
        total_aux_loss = 0

        for block in self.h:
            outputs = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_router_logits=output_router_logits,
                output_aux_loss=output_aux_loss,
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

        return MoeModelOutputWithPastAndAuxLoss(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
            aux_loss=total_aux_loss,
        )

    def _prepare_a_bunch_of_stuff(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        output_router_logits: bool = False,
    ) -> tuple[
        bool,
        bool,
        bool,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor],
    ]:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        return super()._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ) + (output_router_logits,)
