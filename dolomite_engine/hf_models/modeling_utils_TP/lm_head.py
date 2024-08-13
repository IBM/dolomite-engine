import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard

from .embedding import Embedding_TP
from .TP import dtensor_to_tensor, get_module_placements, tensor_to_dtensor


class LMHead_TP(Embedding_TP):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.compute_with_weight(
            input,
            self.weight,
            tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
            use_padding_free_transformer=self.use_padding_free_transformer,
            sequence_parallel=self.sequence_parallel,
        )

    @staticmethod
    def compute_with_weight(
        input: torch.Tensor,
        weight: torch.Tensor,
        tensor_parallel_word_embeddings: bool,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
    ) -> torch.Tensor:
        input = tensor_to_dtensor(
            input, current_placement=get_module_placements(use_padding_free_transformer, sequence_parallel)
        )
        input = F.linear(input, weight)
        input = dtensor_to_tensor(
            input, desired_placement=Shard(-1) if tensor_parallel_word_embeddings else Replicate()
        )
        return input
