import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ....utils import divide_if_divisible
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...mixins import BaseModelMixin, PreTrainedModelMixin
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function
from .cache import RNNCache
from .config import RNNDolomiteConfig
from .layer import RNNDolomiteBlock


class RNNDolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = RNNDolomiteConfig
    layer_class = RNNDolomiteBlock
    _no_split_modules = ["RNNDolomiteBlock"]

    def __init__(self, config: RNNDolomiteConfig, *args, **kwargs):
        self.attention_pattern = config.attention_pattern
        super().__init__(config, *args, **kwargs)

        assert not self._use_padding_free_transformer, "RNN models are not implemented with padding free transformer"


class RNNDolomiteModel(RNNDolomitePreTrainedModel, BaseModelMixin):
    def _init_model(self, config: RNNDolomiteConfig, **kwargs) -> None:
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
    ) -> BaseModelOutputWithPast:
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

        past_key_values = (
            RNNCache(self.attention_pattern) if use_cache and past_key_values is None else past_key_values
        )
        for block in self.h:
            hidden_states = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
