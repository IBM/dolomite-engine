import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ....utils import SafeTensorsWeightsManager
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import RoPE, YaRNScaledRoPE
from ...modeling_utils_TP import Alibi_TP, Dropout_TP, Embedding_TP, get_normalization_function_TP
from ..gpt_dolomite import GPTDolomiteConfig, GPTDolomiteModel, GPTDolomitePreTrainedModel
from .layer import GPTDolomiteBlock_TP


class GPTDolomitePreTrainedModel_TP(GPTDolomitePreTrainedModel):
    _no_split_modules = ["GPTDolomiteBlock_TP"]

    def __init__(self, config: GPTDolomiteConfig, *args, **kwargs):
        GPTDolomitePreTrainedModel.__init__(self, config, *args, **kwargs)

        self.pp_stage = kwargs.get("pp_stage", 0)
        self.num_pp_stages = kwargs.get("num_pp_stages", 1)
        self.is_pp_first_stage = self.pp_stage == 0
        self.is_pp_last_stage = self.pp_stage == self.num_pp_stages - 1

        self.tensor_parallel_word_embeddings = kwargs.get("tensor_parallel_word_embeddings", False)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)


class GPTDolomiteModel_TP(GPTDolomitePreTrainedModel_TP, GPTDolomiteModel):
    def __init__(self, config: GPTDolomiteConfig, **kwargs) -> None:
        GPTDolomitePreTrainedModel_TP.__init__(self, config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.head_dim = self.embed_dim // self.num_heads

        if self.is_pp_first_stage:
            self.wte = Embedding_TP(
                config.vocab_size,
                self.embed_dim,
                std=self.initializer_range,
                tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
                use_padding_free_transformer=self._use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )

            self.drop = nn.Identity() if config.embd_pdrop == 0 else Dropout_TP(config.embd_pdrop)

        layers_per_stage = config.num_hidden_layers // self.num_pp_stages
        start_layer = layers_per_stage * self.pp_stage
        end_layer = min(layers_per_stage * (self.pp_stage + 1), config.num_hidden_layers)

        self.h = nn.ModuleDict(
            {
                str(i): GPTDolomiteBlock_TP(
                    config,
                    normalization_implementation=self.normalization_implementation,
                    attention_implementation=self.attention_implementation,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    layer_idx=i,
                    sequence_parallel=self.sequence_parallel,
                )
                for i in range(start_layer, end_layer)
            }
        )

        if self.is_pp_last_stage:
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

    def get_input_embeddings(self) -> Embedding_TP:
        return self.wte

    def set_input_embeddings(self, new_embeddings: Embedding_TP) -> None:
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
        if self.num_pp_stages == 1:
            output = super().forward(
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
        else:
            if self.is_pp_first_stage:
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
            else:
                hidden_states, attention_mask, rope_cos_sin, cu_seqlens, max_seqlen = (
                    input_ids,
                    past_key_values,
                    attention_mask,
                    token_type_ids,
                    position_ids,
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

            if self.is_pp_last_stage:
                hidden_states = self.ln_f(hidden_states)

            output = hidden_states, attention_mask, rope_cos_sin, cu_seqlens, max_seqlen

        return output

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        # word embeddings
        self.wte.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "wte.")

        # positional embeddings
        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.wpe.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "wpe.")
        elif self.position_embedding_type == PositionEmbeddingType.alibi:
            with torch.device(torch.cuda.current_device()):
                self.alibi.reset_parameters()
        elif self.position_embedding_type == PositionEmbeddingType.rope:
            with torch.device(torch.cuda.current_device()):
                self.rope.reset_parameters()
        else:
            raise ValueError(f"unexpected position_embedding_type ({self.position_embedding_type})")

        # layers
        for layer_idx, block in tqdm(enumerate(self.h), desc="Loading layers"):
            block.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + f"h.{layer_idx}.")

        # final layernorm
        state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "ln_f.weight")}
        if hasattr(self.ln_f, "bias"):
            state_dict["bias"] = safetensors_weight_manager.get_tensor(prefix + "ln_f.bias")
        self.ln_f.load_state_dict(state_dict)

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            if self.is_pp_first_stage:
                self.wpe = Embedding_TP(
                    max_position_embeddings,
                    self.embed_dim,
                    std=self.initializer_range,
                    tensor_parallel_word_embeddings=False,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    sequence_parallel=self.sequence_parallel,
                )
        elif self.position_embedding_type == PositionEmbeddingType.alibi:
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
