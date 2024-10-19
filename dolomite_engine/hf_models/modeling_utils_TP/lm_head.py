import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

from .embedding import Embedding_TP
from .TP import dtensor_to_tensor, get_module_placements, tensor_to_dtensor


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

        if torch._inductor.config._micro_pipeline_tp:
            self.compile()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.compute_with_weight(
            input,
            self.weight,
            tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
            use_padding_free_transformer=self.use_padding_free_transformer,
            sequence_parallel=self.sequence_parallel,
            tp_mesh=self.tp_mesh,
        )

    @staticmethod
    def compute_with_weight(
        input: torch.Tensor,
        weight: torch.Tensor,
        tensor_parallel_word_embeddings: bool,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
        tp_mesh: DeviceMesh,
    ) -> torch.Tensor:
        input = tensor_to_dtensor(
            input,
            device_mesh=tp_mesh,
            current_placement=get_module_placements(use_padding_free_transformer, sequence_parallel),
        )
        input = F.linear(input, weight)
        input = dtensor_to_tensor(
            input, device_mesh=tp_mesh, desired_placement=Shard(-1) if tensor_parallel_word_embeddings else Replicate()
        )
        return input
