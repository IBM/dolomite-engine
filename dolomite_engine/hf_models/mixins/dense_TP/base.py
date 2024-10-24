import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ....utils import ProcessGroupManager, divide_if_divisible
from ...config import CommonConfig
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import RoPE, YaRNScaledRoPE
from ...modeling_utils_TP import Alibi_TP, Dropout_TP, Embedding_TP, get_normalization_function_TP
from ..dense import BaseModelMixin, PreTrainedModelMixin


class PreTrainedModelMixin_TP(PreTrainedModelMixin):
    def __init__(self, config: CommonConfig, *args, **kwargs) -> None:
        self.tensor_parallel_word_embeddings = kwargs.get("tensor_parallel_word_embeddings", False)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)

        self.num_pipeline_stages = kwargs.get("num_pipeline_stages", 1)
        self.pipeline_stage_id = kwargs.get("pipeline_stage_id", 0)

        self.is_first_stage = self.pipeline_stage_id == 0
        self.is_last_stage = self.pipeline_stage_id == self.num_pipeline_stages - 1
        self.is_pipeline_parallel_enabled = self.num_pipeline_stages > 1

        super().__init__(config, *args, **kwargs)

        if self.is_pipeline_parallel_enabled and self._tied_word_embeddings:
            raise NotImplementedError()


class BaseModelMixin_TP(PreTrainedModelMixin_TP, BaseModelMixin):
    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.head_dim = self.embed_dim // self.num_heads

        self.layers_per_stage = divide_if_divisible(
            config.n_layer, self.num_pipeline_stages, "layers should be divisible by num_pipeline_stages"
        )

        self.layer_start_id = self.layers_per_stage * self.pipeline_stage_id
        self.layer_end_id = self.layers_per_stage * (self.pipeline_stage_id + 1)

        if self.is_first_stage:
            self.wte = Embedding_TP(
                config.vocab_size,
                self.embed_dim,
                std=self.initializer_range,
                tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
                use_padding_free_transformer=self._use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )

            self.drop = (
                nn.Identity()
                if config.embd_pdrop == 0
                else Dropout_TP(
                    config.embd_pdrop,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    sequence_parallel=self.sequence_parallel,
                )
            )

        self.h = nn.ModuleDict(
            {
                str(i): self.layer_class(
                    config,
                    normalization_implementation=self.normalization_implementation,
                    attention_implementation=self.attention_implementation,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    layer_idx=i,
                    sequence_parallel=self.sequence_parallel,
                )
                for i in range(self.layer_start_id, self.layer_end_id)
            }
        )

        if self.is_last_stage:
            self.ln_f = get_normalization_function_TP(
                config.normalization_function,
                self.embed_dim,
                eps=config.layer_norm_epsilon,
                normalization_implementation=self.normalization_implementation,
                use_padding_free_transformer=self._use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
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
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        if self.is_first_stage:
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
        else:
            assert past_key_values is None
            assert attention_mask is None

            hidden_states = input_ids
            past_length = 0

            if self._use_padding_free_transformer:
                key_length = max_seqlen
                # query length will change if past_key_values is not None
                query_length = key_length - past_length
            else:
                key_length = (
                    hidden_states.size(1) * ProcessGroupManager.get_tensor_parallel_world_size()
                    if self.sequence_parallel
                    else hidden_states.size(1)
                )
                query_length = key_length - past_length

            position_ids = torch.arange(past_length, key_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

            rope_cos_sin = self._get_rope_cos_sin(
                key_length, position_ids, dtype=hidden_states.dtype, device=hidden_states.device
            )

        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        all_hidden_states = () if output_hidden_states else None

        for layer_idx in range(self.layer_start_id, self.layer_end_id):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = self.h[str(layer_idx)](
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        if self.is_last_stage:
            hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            if self.is_first_stage:
                self.wpe = Embedding_TP(
                    max_position_embeddings,
                    self.embed_dim,
                    std=self.initializer_range,
                    tensor_parallel_word_embeddings=False,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    sequence_parallel=self.sequence_parallel,
                )
        elif self.position_embedding_type == PositionEmbeddingType.alibi:
            if self.is_pipeline_parallel_enabled:
                raise NotImplementedError()

            self.alibi = Alibi_TP(self.num_heads)
        elif self.position_embedding_type == PositionEmbeddingType.rope:
            if self.config.rope_scaling is None:
                self.rope = RoPE(
                    self.head_dim, max_position_embeddings=max_position_embeddings, base=self.config.rope_theta
                )
            else:
                self.rope = YaRNScaledRoPE(
                    self.head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                    scale=self.config.rope_scaling["factor"],
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
        else:
            raise NotImplementedError()
