import torch
import torch.nn as nn
from transformers import Cache

from ....utils import divide_if_divisible
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...mixins import BaseMoEModelMixin, MoeModelOutputWithPastAndAuxLoss, PreTrainedMoEModelMixin
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function
from ..rnn_dolomite.base import RNNDolomiteModel, RNNDolomitePreTrainedModel
from ..rnn_dolomite.cache import RNNCache
from .config import RNNMoEDolomiteConfig
from .layer import RNNMoEDolomiteBlock


class RNNMoEDolomitePreTrainedModel(PreTrainedMoEModelMixin, RNNDolomitePreTrainedModel):
    config_class = RNNMoEDolomiteConfig
    layer_class = RNNMoEDolomiteBlock
    _no_split_modules = ["RNNMoEDolomiteBlock"]


class RNNMoEDolomiteModel(RNNMoEDolomitePreTrainedModel, BaseMoEModelMixin, RNNDolomiteModel):
    def _init_model(self, config: RNNMoEDolomiteConfig, **kwargs) -> None:
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
                    attention_pattern=self.attention_pattern[i],
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
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> MoeModelOutputWithPastAndAuxLoss:
        (
            use_cache,
            hidden_states,
            causal_mask,
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

        past_key_values = (
            RNNCache(self.attention_pattern) if use_cache and past_key_values is None else past_key_values
        )
        total_aux_loss = 0

        for block in self.h:
            hidden_states, aux_loss = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            total_aux_loss = total_aux_loss + aux_loss

        hidden_states = self.ln_f(hidden_states)

        return MoeModelOutputWithPastAndAuxLoss(
            last_hidden_state=hidden_states, past_key_values=past_key_values, aux_loss=total_aux_loss
        )
