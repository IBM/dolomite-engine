import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard

from ...utils import is_dtensors_computation_enabled
from .embedding import Embedding_TP
from .TP import copy_to_tensor_parallel_region, dtensor_to_tensor, tensor_to_dtensor


class LMHead_TP(Embedding_TP):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.compute_with_weight(input, self.weight, self.tensor_parallel_word_embeddings)

    @staticmethod
    def compute_with_weight(
        input: torch.Tensor, weight: torch.Tensor, tensor_parallel_word_embeddings: bool
    ) -> torch.Tensor:
        if is_dtensors_computation_enabled():
            input = tensor_to_dtensor(input, current_placement=Replicate())
            input = F.linear(input, weight)
            input = dtensor_to_tensor(
                input, desired_placement=Shard(-1) if tensor_parallel_word_embeddings else Replicate()
            )
        else:
            if tensor_parallel_word_embeddings:
                input = copy_to_tensor_parallel_region(input)

            input = F.linear(input, weight.to_local())

        return input
