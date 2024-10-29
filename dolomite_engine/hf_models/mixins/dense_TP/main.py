from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from ....utils import ProcessGroupManager, SafeTensorsWeightsManager, divide_if_divisible
from ...config import CommonConfig
from ...enums import PositionEmbeddingType
from ...modeling_utils_TP import LMHead_TP, dtensor_to_tensor, tensor_to_dtensor
from ..dense import CausalLMModelMixin
from .base import PreTrainedModelMixin_TP


class CausalLMModelMixin_TP(PreTrainedModelMixin_TP, CausalLMModelMixin):
    model_parallel_state_dict_function = None

    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.vocab_size = config.vocab_size
        self.transformer = self.base_model_class(config, **kwargs)

        if self.is_last_stage:
            if not self._tied_word_embeddings:
                self.lm_head = LMHead_TP(
                    self.vocab_size,
                    config.n_embd,
                    std=config.initializer_range,
                    tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
                    sequence_parallel=self.sequence_parallel,
                )

            self.m_width = config.m_width
            self.upcast_logits_for_loss = config.upcast_logits_for_loss

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        # Initialize weights and apply final processing
        self.post_init()

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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        output_parallel_lm_logits: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast | torch.Tensor:
        if not self.is_pipeline_parallel_enabled or self.is_first_stage:
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
                output_attentions=output_attentions,
            )

        transformer_outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
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

        if not self.is_pipeline_parallel_enabled or self.is_last_stage:
            lm_logits = self.get_lm_logits(transformer_outputs.last_hidden_state)

            if self.m_width is not None:
                lm_logits = lm_logits / self.m_width

        if not self.is_pipeline_parallel_enabled:
            loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels, cu_seqlens)

        if not self.is_pipeline_parallel_enabled or self.is_last_stage:
            if output_parallel_lm_logits:
                assert self.tensor_parallel_word_embeddings
            else:
                if self.tensor_parallel_word_embeddings:
                    # all gather
                    lm_logits = tensor_to_dtensor(lm_logits, device_mesh=self.tp_mesh, current_placement=Shard(-1))
                    lm_logits = dtensor_to_tensor(lm_logits, device_mesh=self.tp_mesh, desired_placement=Replicate())

        if not self.is_pipeline_parallel_enabled:
            output = CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        elif self.is_last_stage:
            output = lm_logits
        else:
            output = transformer_outputs.last_hidden_state

        return output

    def get_lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (
            LMHead_TP.compute_with_weight(
                hidden_states,
                weight=self.transformer.wte.weight,
                tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
                use_padding_free_transformer=self._use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
                tp_mesh=self.tp_mesh,
            )
            if self._tied_word_embeddings
            else self.lm_head(hidden_states)
        )

    def get_autoregressive_language_modeling_loss(
        self, lm_logits: torch.Tensor, labels: torch.Tensor | None, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        if labels is None:
            return None

        if self._use_padding_free_transformer:
            shift_logits = lm_logits[:-1, :]
            shift_labels = labels[1:].to(shift_logits.device)

            # this is needed so that the last token of current example doesn't predict first token of next example
            drop_loss_positions = cu_seqlens[1:-1] - 1
            shift_labels[drop_loss_positions] = -100
        else:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

        shift_logits = tensor_to_dtensor(
            shift_logits,
            device_mesh=self.tp_mesh,
            current_placement=Shard(-1) if self.tensor_parallel_word_embeddings else Replicate(),
        )
        shift_labels = tensor_to_dtensor(shift_labels, device_mesh=self.tp_mesh, current_placement=Replicate())

        if self.upcast_logits_for_loss:
            shift_logits = shift_logits.float()

        loss_context = loss_parallel if self.tensor_parallel_word_embeddings else nullcontext
        with loss_context():
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float32,
        tensor_parallel_word_embeddings: bool = False,
        **kwargs,
    ) -> CausalLMModelMixin_TP:
        config: CommonConfig = cls.config_class.from_pretrained(pretrained_model_name_or_path)

        # use dummy tensors to avoid initializing model here
        with torch.device("meta"):
            # try sharding vocab matrices if really struggling for memory
            model = cls._from_config(config, tensor_parallel_word_embeddings=tensor_parallel_word_embeddings, **kwargs)
            model = model.to(dtype=torch_dtype)

        # copy to device without copying storage
        model = model.to_empty(device=torch.cuda.current_device())
        model.load_from_safetensors_weights_manager(SafeTensorsWeightsManager(pretrained_model_name_or_path))

        return model

    def load_from_safetensors_weights_manager(self, safetensors_weights_manager: SafeTensorsWeightsManager) -> None:
        with torch.device(torch.cuda.current_device()):
            position_embedding_type = PositionEmbeddingType(self.config.position_embedding_type)

            if position_embedding_type == PositionEmbeddingType.alibi:
                self.transformer.alibi.reset_parameters()
            elif position_embedding_type == PositionEmbeddingType.rope:
                self.transformer.rope.reset_parameters()

        state_dict = self.__class__.model_parallel_state_dict_function(
            config=self.config,
            safetensors_weights_manager=safetensors_weights_manager,
            tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
            num_pipeline_stages=self.num_pipeline_stages,
            pipeline_stage_id=self.pipeline_stage_id,
        )

        self.load_state_dict(state_dict)

    def get_dummy_input_tensor(
        self, micro_batch_size: int, sequence_length: int, intermediate_dtype: torch.dtype
    ) -> tuple[int]:
        if self.is_first_stage:
            # 1 is added to sequence length since megatron's dataloader gives an extra token and for good reason
            tensor = torch.empty(
                micro_batch_size, sequence_length + 1, device=torch.cuda.current_device(), dtype=torch.long
            )
        else:
            tensor = self._get_dummy_intermediate_tensor(
                micro_batch_size, sequence_length, intermediate_dtype=intermediate_dtype
            )

        return tensor

    def get_dummy_output_tensor(
        self,
        micro_batch_size: int,
        sequence_length: int,
        intermediate_dtype: torch.dtype,
        output_parallel_lm_logits_if_possible: bool,
    ) -> tuple[int]:
        if self.is_last_stage:
            vocab_size = self.config.vocab_size
            if self.tensor_parallel_word_embeddings and output_parallel_lm_logits_if_possible:
                vocab_size = divide_if_divisible(vocab_size, ProcessGroupManager.get_tensor_parallel_world_size(), "")

            if self._use_padding_free_transformer:
                tensor = torch.empty(
                    micro_batch_size * sequence_length,
                    vocab_size,
                    device=torch.cuda.current_device(),
                    dtype=intermediate_dtype,
                )
            else:
                tensor = torch.empty(
                    micro_batch_size,
                    sequence_length,
                    vocab_size,
                    device=torch.cuda.current_device(),
                    dtype=intermediate_dtype,
                )
        else:
            tensor = self._get_dummy_intermediate_tensor(
                micro_batch_size, sequence_length, intermediate_dtype=intermediate_dtype
            )

        return tensor

    def _get_dummy_intermediate_tensor(
        self, micro_batch_size: int, sequence_length: int, intermediate_dtype: torch.dtype
    ) -> tuple[int]:
        sharded_sequence_length = (
            divide_if_divisible(sequence_length, ProcessGroupManager.get_tensor_parallel_world_size(), "")
            if self.sequence_parallel
            else sequence_length
        )

        hidden_size = self.config.hidden_size

        if self._use_padding_free_transformer:
            tensor = torch.empty(
                micro_batch_size * sharded_sequence_length,
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=intermediate_dtype,
            )
        else:
            tensor = torch.empty(
                micro_batch_size,
                sharded_sequence_length,
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=intermediate_dtype,
            )

        return tensor
