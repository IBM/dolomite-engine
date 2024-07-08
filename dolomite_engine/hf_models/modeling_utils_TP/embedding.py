import math
from contextlib import nullcontext
from functools import partial
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard

from ...utils import ProcessGroupManager, SafeTensorsWeightsManager, get_cuda_rng_tracker
from ..modeling_utils import ParameterizedEmbedding
from ..utils import divide_if_divisible
from .TP import dtensor_to_tensor_hook, modify_state_dict_to_dtensor_dict, tensor_to_dtensor_hook


class Embedding_TP(ParameterizedEmbedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, std: float = None, tensor_parallel_word_embeddings: bool = False
    ) -> None:
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.tensor_parallel_word_embeddings = tensor_parallel_word_embeddings and self.tp_world_size > 1

        if self.tensor_parallel_word_embeddings:
            self.vocab_start_index, self.vocab_end_index, num_embeddings_per_tp_rank = get_tensor_parallel_vocab_info(
                num_embeddings
            )

            super().__init__(num_embeddings_per_tp_rank, embedding_dim, std=std)

            placement = Shard(0)
        else:
            with get_cuda_rng_tracker().fork():
                super().__init__(num_embeddings, embedding_dim, std=std)

            placement = Replicate()

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[placement]
            )
        )

        self.register_forward_pre_hook(partial(tensor_to_dtensor_hook, current_placement=Replicate()))
        self.register_forward_hook(partial(dtensor_to_tensor_hook, desired_placement=Replicate()))

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        if self.tensor_parallel_word_embeddings:
            weight = safetensors_weight_manager.get_slice(prefix + "weight")[
                self.vocab_start_index : self.vocab_end_index, :
            ]
            if self.num_embeddings > weight.shape[0]:
                weight = torch.cat(
                    [
                        weight,
                        torch.zeros((self.num_embeddings - weight.shape[0], weight.shape[1])),
                    ]
                )
        else:
            weight = safetensors_weight_manager.get_tensor(prefix + "weight")

        self.load_state_dict({"weight": weight})

    def reset_parameters(self) -> None:
        context = nullcontext if self.tensor_parallel_word_embeddings else get_cuda_rng_tracker().fork

        with context():
            return super().reset_parameters()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)


def get_tensor_parallel_vocab_info(vocab_size: int, make_vocab_size_divisible_by: int = 64) -> tuple[int, int, int]:
    tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

    divide_if_divisible(make_vocab_size_divisible_by, tp_world_size, "")

    vocab_size_per_tensor_parallel_rank = (
        make_vocab_size_divisible_by * math.ceil(vocab_size / make_vocab_size_divisible_by)
    ) // tp_world_size

    vocab_start_index = tp_rank * vocab_size_per_tensor_parallel_rank
    vocab_end_index = min((tp_rank + 1) * vocab_size_per_tensor_parallel_rank, vocab_size)

    return vocab_start_index, vocab_end_index, vocab_size_per_tensor_parallel_rank
