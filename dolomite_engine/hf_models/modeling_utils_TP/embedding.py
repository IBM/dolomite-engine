import torch
import torch.nn as nn

from ...utils import SafeTensorsWeightsManager
from .TP import get_tensor_parallel_group_manager, reduce_from_tensor_parallel_region


class Embedding_TP(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, make_vocab_size_divisible_by: int = 64) -> None:
        self.tp_rank = get_tensor_parallel_group_manager().get_rank()
        self.tp_world_size = get_tensor_parallel_group_manager().get_world_size()

        assert make_vocab_size_divisible_by % self.tp_world_size == 0

        embedding_matrix_size_per_tp_rank = (
            make_vocab_size_divisible_by * (1 + (num_embeddings // make_vocab_size_divisible_by))
        ) // self.tp_world_size

        self.vocab_start_index = self.tp_rank * embedding_matrix_size_per_tp_rank
        self.vocab_end_index = min((self.tp_rank + 1) * embedding_matrix_size_per_tp_rank, num_embeddings)

        super().__init__(embedding_matrix_size_per_tp_rank, embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tp_world_size > 1:
            # Build the mask.
            input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)
            # Mask the input.
            masked_input = input.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input

        output_parallel = super().forward(masked_input)

        # Mask the output embedding.
        if self.tp_world_size > 1:
            output_parallel[input_mask, :] = 0

        return reduce_from_tensor_parallel_region(output_parallel)

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
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

        self.load_state_dict({"weight": weight})
