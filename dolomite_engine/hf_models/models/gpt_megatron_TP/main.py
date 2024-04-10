from __future__ import annotations

from typing import List, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from ...modeling_utils import ParameterizedLinear
from ...safetensors import SafeTensorsWeightsManager
from ..gpt_megatron import GPTMegatronConfig, GPTMegatronForCausalLM, GPTMegatronPreTrainedModel
from .base import GPTMegatronModel_TP


class GPTMegatronForCausalLM_TP(GPTMegatronForCausalLM):
    def __init__(
        self,
        config: GPTMegatronConfig,
        tensor_parallel_vocab_matrix: bool = False,
        tensor_parallel_position_embedding_matrix: bool = False,
        **kwargs,
    ) -> None:
        GPTMegatronPreTrainedModel.__init__(self, config, **kwargs)

        self.tensor_parallel_vocab_matrix = tensor_parallel_vocab_matrix

        self.transformer = GPTMegatronModel_TP(
            config,
            tensor_parallel_vocab_matrix=tensor_parallel_vocab_matrix,
            tensor_parallel_position_embedding_matrix=tensor_parallel_position_embedding_matrix,
            **kwargs,
        )

        # we don't split lm_head for now
        # TODO investigate how to split this for HF generate API
        self.lm_head = ParameterizedLinear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        if not self.tensor_parallel_vocab_matrix:
            self.tie_weights()

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[int]] = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: Union[torch.Tensor, List[int]] = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        if labels is not None:
            # TODO investigate how to implement VocabParallelCrossEntropy and how to keep weights tied
            # we can also drop the lm_head entirely if that works
            # also note that the weights are not tied together when voacb matrix is split across TP processes
            # need a better way to tie them
            raise NotImplementedError()

        return super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
        self.transformer.load_unsharded_weights(safetensors_weight_manager, prefix + "transformer.")

        if self.tensor_parallel_vocab_matrix:
            state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "transformer.wte.weight")}
            self.lm_head.load_state_dict(state_dict)
        else:
            self.post_init()

    @classmethod
    def from_pretrained(
        self,
        model_name: str,
        torch_dtype: torch.dtype = torch.float32,
        tensor_parallel_vocab_matrix: bool = False,
        tensor_parallel_position_embedding_matrix: bool = False,
    ) -> GPTMegatronForCausalLM_TP:
        # use dummy tensors to avoid initializing model here
        with torch.device("meta"):
            config = GPTMegatronConfig.from_pretrained(model_name)
            # try sharding vocab matrices if really struggling for memory
            model = GPTMegatronForCausalLM_TP(
                config,
                tensor_parallel_vocab_matrix=tensor_parallel_vocab_matrix,
                tensor_parallel_position_embedding_matrix=tensor_parallel_position_embedding_matrix,
            )

            # set dtype
            model = model.to(dtype=torch_dtype)

        # copy to device without copying storage
        model = model.to_empty(device=torch.cuda.current_device())

        # load weights into tensor parallel model using SafeTensorsWeightsManager class
        # this avoids loading multiple copies of the parameters in CPU memory
        safetensors_weight_manager = SafeTensorsWeightsManager(model_name)
        model.load_unsharded_weights(safetensors_weight_manager)

        # TODO call self.post_init() for non-TP vocab matrix here

        return model
