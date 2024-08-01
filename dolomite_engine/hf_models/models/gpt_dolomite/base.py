import warnings

import torch
import torch.nn as nn
from transformers import DynamicCache, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...defaults import DEFAULT_NORMALIZATION_IMPLEMENTATION
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import Alibi, ParameterizedEmbedding, RMSNorm, RoPE, YaRNScaledRoPE, get_normalization_function
from ...utils import convert_padding_free_lists_to_tensors, divide_if_divisible
from .config import GPTDolomiteConfig
from .layer import GPTDolomiteBlock


class GPTDolomitePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTDolomiteConfig
    base_model_prefix = "transformer"
    causal = True
    _no_split_modules = ["GPTDolomiteBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config: GPTDolomiteConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.normalization_implementation = kwargs.get(
            "normalization_implementation", DEFAULT_NORMALIZATION_IMPLEMENTATION
        )

        self.attention_implementation = self.config._attn_implementation
        self._use_eager_attention = self.attention_implementation == "eager"
        self._use_sdpa = self.attention_implementation == "sdpa"
        self._use_flash_attention_2 = self.attention_implementation == "flash_attention_2"
        self._use_padding_free_transformer = kwargs.get("use_padding_free_transformer", False)

        self._tied_word_embeddings = config.tie_word_embeddings

        if self._use_padding_free_transformer:
            assert self._use_flash_attention_2, "padding free transformer only works with flash attention"

        assert any(
            [
                self._use_eager_attention,
                self._use_sdpa,
                self._use_flash_attention_2,
                self._use_padding_free_transformer,
            ]
        ) and not all(
            [
                self._use_eager_attention,
                self._use_sdpa,
                self._use_flash_attention_2,
                self._use_padding_free_transformer,
            ]
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear, nn.LayerNorm, RMSNorm, Alibi, RoPE)):
            module.reset_parameters()

    def prepare_inputs_for_model(
        self,
        input_ids: torch.Tensor | list[list[int]] | None,
        inputs_embeds: torch.Tensor | list[list[float]] | None,
        position_ids: torch.Tensor | list[list[int]] | None,
        token_type_ids: torch.Tensor | list[list[int]] | None,
        labels: torch.Tensor | list[list[int]] | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | None,
        past_key_values: tuple[tuple[torch.Tensor]],
        attention_mask: torch.Tensor | None,
        use_cache: bool,
        output_attentions: bool,
    ) -> tuple[torch.Tensor]:
        if self._use_padding_free_transformer:
            if isinstance(input_ids, list) or isinstance(inputs_embeds, list):
                # this is managed internally
                error_message = (
                    "{variable} should not be passed for flash attention when using List[List[int]] "
                    "input types attention mask logic is handled internally"
                )
                assert cu_seqlens is None, error_message.format(variable="cu_seqlens")
                assert max_seqlen is None, error_message.format(variable="max_seqlen")
                assert attention_mask is None, error_message.format(variable="attention_mask")

                input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen = (
                    convert_padding_free_lists_to_tensors(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )
                )
            else:
                assert (
                    cu_seqlens is not None
                ), "cu_seqlens needs to be specified when using tensor inputs with padding_free transformer"
                assert position_ids is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
                assert max_seqlen is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
                assert attention_mask is None, "attention_mask should not be passed when specifying cu_seqlens"

            if use_cache or past_key_values is not None:
                raise NotImplementedError("KV caching is not supported with padding_free transformer")

        assert not output_attentions

        return input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen


class GPTDolomiteModel(GPTDolomitePreTrainedModel):
    mask_value = None

    def __init__(self, config: GPTDolomiteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
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
                GPTDolomiteBlock(
                    config,
                    normalization_implementation=self.normalization_implementation,
                    attention_implementation=self.attention_implementation,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    layer_idx=i,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = get_normalization_function(
            config.normalization_function,
            self.embed_dim,
            eps=config.layer_norm_epsilon,
            normalization_implementation=self.normalization_implementation,
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ParameterizedEmbedding:
        return self.wte

    def set_input_embeddings(self, new_embeddings: ParameterizedEmbedding) -> None:
        self.wte = new_embeddings

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

        # ==========================================================================================
        # padding_free:
        #     attention_mask -> None
        # flash:
        #     attention_mask -> (batch_size, key_length)
        # else:
        #     attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

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

    def _get_position_ids(
        self, attention_mask: torch.Tensor, past_length: int, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        if attention_mask is not None and len(attention_mask.shape) == 2:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_length > 0:
                position_ids = position_ids[:, past_length:key_length:]
        else:
            position_ids = torch.arange(past_length, key_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

        return position_ids

    def _get_alibi_bias(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.position_embedding_type != PositionEmbeddingType.alibi:
            return None

        alibi_bias = self.alibi(attention_mask, batch_size, key_length, device, dtype)

        # ==========================================================================================
        # alibi_bias -> (batch_size, num_heads, key_length)
        # ==========================================================================================

        alibi_bias = alibi_bias.unsqueeze(2)
        if query_length != 1:
            alibi_bias = alibi_bias.expand(-1, -1, query_length, -1)

        # ==========================================================================================
        # alibi_bias -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        return alibi_bias

    def _get_rope_cos_sin(
        self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if self.position_embedding_type == PositionEmbeddingType.rope:
            cos, sin = self.rope(key_length, dtype=dtype, device=device)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
            return cos, sin

    def _prepare_causal_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        past_length = key_length - query_length

        # ==========================================================================================
        # attention_mask -> (batch_size, key_length)
        # ==========================================================================================

        if query_length > 1:
            # (query_length, key_length)
            causal_mask = torch.empty((query_length, key_length), dtype=torch.bool, device=device)
            causal_mask[:, past_length:] = torch.tril(
                torch.ones(query_length, query_length, dtype=torch.bool, device=device)
            )

            if past_length > 0:
                causal_mask[:, :past_length] = True

            # (query_length, key_length) -> (1, query_length, key_length)
            causal_mask = causal_mask.unsqueeze(0)

            if attention_mask is None:
                # (1, query_length, key_length) -> (batch_size, query_length, key_length)
                causal_mask = causal_mask.expand(batch_size, -1, -1)
            else:
                # (1, query_length, key_length) & (batch_size, 1, key_length) -> (batch_size, query_length, key_length)
                causal_mask = causal_mask & attention_mask.unsqueeze(1).to(torch.bool)
        else:
            if attention_mask is None:
                # (batch_size, query_length, key_length)
                causal_mask = torch.ones(batch_size, query_length, key_length, dtype=torch.bool, device=device)
            else:
                # (batch_size, query_length, key_length)
                causal_mask = attention_mask.unsqueeze(1).to(dtype=torch.bool, device=device)

        # ==========================================================================================
        # attention_mask -> (batch_size, query_length, key_length)
        # ==========================================================================================

        causal_mask = causal_mask.unsqueeze(1)

        # ==========================================================================================
        # attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        return causal_mask

    def _get_initial_hidden_state(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            inputs_embeds = inputs_embeds + self.wpe(position_ids)

        if token_type_ids is not None:
            inputs_embeds = inputs_embeds + self.wte(token_type_ids)

        inputs_embeds = self.drop(inputs_embeds)

        if self.m_emb is not None:
            inputs_embeds = inputs_embeds * self.m_emb

        return inputs_embeds

    def _prepare_a_bunch_of_stuff(
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if use_cache is None:
            use_cache = False if self._use_padding_free_transformer else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()

            # special handling for padding free transformer with list inputs
            if self._use_padding_free_transformer:
                # for flash attention, there is no padding and we do packing
                # so, input_ids is of shape (s1 + s2 + ... + sb)
                batch_size = cu_seqlens.shape[0] - 1
            else:
                batch_size = input_shape[0]
        elif inputs_embeds is not None:
            # TODO special handling for padding free transformer needed here if we support inputs_embeds argument
            input_shape = inputs_embeds.size()[:-1]
            batch_size = input_shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if self._use_padding_free_transformer:
            assert position_ids is not None, (
                "GPTDolomiteModel needs position_ids from outside when using flash attention with List[List[int]] "
                "inputs"
            )
        else:
            if self.position_embedding_type == PositionEmbeddingType.alibi:
                if position_ids is not None:
                    warnings.warn("`position_ids` have no functionality with Alibi.", FutureWarning)

            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])

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

        past_length = None
        query_length = None
        key_length = None
        if self._use_padding_free_transformer:
            key_length = max_seqlen
        else:
            past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            query_length = input_shape[-1]
            key_length = past_length + query_length

        if position_ids is None:
            position_ids = self._get_position_ids(attention_mask, past_length, query_length, key_length, device)

        # ==========================================================================================
        # padding_free:
        #     input_ids -> (total_q)
        #     attention_mask -> None
        #     position_ids -> (total_q)
        # else:
        #     input_ids -> (batch_size, query_length)
        #     attention_mask -> None or (batch_size, key_length)
        #     position_ids -> (batch_size, query_length)
        # ==========================================================================================

        hidden_states = self._get_initial_hidden_state(input_ids, inputs_embeds, position_ids, token_type_ids)

        # ==========================================================================================
        # padding_free:
        #     hidden_states -> (total_q, num_heads * head_dim)
        # else:
        #     hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        alibi_bias = self._get_alibi_bias(
            attention_mask, batch_size, query_length, key_length, device, hidden_states.dtype
        )

        # ==========================================================================================
        # alibi_bias -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        rope_cos_sin = self._get_rope_cos_sin(
            key_length, position_ids, dtype=hidden_states.dtype, device=hidden_states.device
        )

        # ==========================================================================================
        # padding_free:
        #     rope_cos_sin -> 2 * (max_seqlen, head_dim)
        # else:
        #     rope_cos_sin -> 2 * (key_length, head_dim)
        # ==========================================================================================

        attention_mask = self._get_maybe_causal_mask(
            attention_mask, alibi_bias, batch_size, query_length, key_length, hidden_states.dtype, device
        )

        return (
            output_hidden_states,
            use_cache,
            return_dict,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
        )

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.wpe = ParameterizedEmbedding(max_position_embeddings, self.embed_dim, std=self.initializer_range)
        elif self.position_embedding_type == PositionEmbeddingType.alibi:
            assert not self._use_flash_attention_2, "alibi is not implemented with FlashAttention"

            self.alibi = Alibi(self.num_heads)
        elif self.position_embedding_type == PositionEmbeddingType.rope:
            if self.config.rope_scaling is None:
                self.rope = RoPE(
                    self.head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                )
            else:
                self.rope = YaRNScaledRoPE(
                    self.head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                    scale=self.config.rope_scaling["factor"],
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
        elif self.position_embedding_type == PositionEmbeddingType.nope:
            pass
        else:
            raise NotImplementedError()

    def _get_mask_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    def _get_maybe_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        alibi_bias: torch.Tensor | None,
        batch_size: int,
        query_length: int,
        key_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self._use_sdpa:
            # we use the causal/non-causal argument of SDPA for attention in this case
            if attention_mask is not None:
                attention_mask = self._prepare_causal_attention_mask(
                    attention_mask, batch_size, query_length, key_length, device
                )

                attention_mask = torch.where(
                    attention_mask,
                    ~attention_mask if alibi_bias is None else alibi_bias,
                    self._get_mask_value(attention_mask.device, dtype),
                )

                # this is needed to prevent NaN since SDPA
                # see issue: https://github.com/pytorch/pytorch/issues/110213
                attention_mask = attention_mask * ~torch.all(
                    attention_mask == self._get_mask_value(attention_mask.device, dtype), dim=-1, keepdim=True
                )
        elif self._use_eager_attention:
            attention_mask = self._prepare_causal_attention_mask(
                attention_mask, batch_size, query_length, key_length, device
            )

            attention_mask = torch.where(
                attention_mask,
                ~attention_mask if alibi_bias is None else alibi_bias,
                self._get_mask_value(attention_mask.device, dtype),
            )

        return attention_mask
