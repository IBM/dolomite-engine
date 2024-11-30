from ...config import CommonConfig
from ...enums import PositionEmbeddingType


class GPTEnsembleConfig(CommonConfig):
    model_type = "gpt_ensemble"

    def __init__(
        self,
        pretraining_tensor_parallel_size: int = 1,
        reduce_pattern: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.pretraining_tensor_parallel_size = pretraining_tensor_parallel_size

        if PositionEmbeddingType(self.position_embedding_type) == PositionEmbeddingType.alibi:
            raise NotImplementedError("currently GPTEnsemble doesn't support alibi")

        self.reduce_pattern = (
            [{"attention": True, "mlp": True} for i in range(self.n_layer)]
            if reduce_pattern is None
            else reduce_pattern
        )
