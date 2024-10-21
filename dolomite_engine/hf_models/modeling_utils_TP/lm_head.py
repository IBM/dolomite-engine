import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard

from .embedding import Embedding_TP
from .TP import dtensor_to_tensor, tensor_to_dtensor, use_async_tensor_parallel


class LMHead_TP(Embedding_TP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        std: float | None = None,
        tensor_parallel_word_embeddings: bool = False,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            std,
            tensor_parallel_word_embeddings,
            use_padding_free_transformer,
            sequence_parallel,
        )

        if use_async_tensor_parallel():
            self.compile()

    def forward(self, input: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=self.output_placement)
        input = F.linear(input, self.weight if weight is None else weight)
        input = dtensor_to_tensor(
            input,
            device_mesh=self.tp_mesh,
            desired_placement=Shard(-1) if self.tensor_parallel_word_embeddings else Replicate(),
        )
        return input
