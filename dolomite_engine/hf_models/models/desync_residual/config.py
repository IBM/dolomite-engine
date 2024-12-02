from ...config import CommonConfig
from ...enums import PositionEmbeddingType


class DesyncResidualConfig(CommonConfig):
    model_type = "desync_residual"

    def __init__(
        self,
        pretraining_tensor_parallel_size: int = 1,
        reduce_pattern: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.pretraining_tensor_parallel_size = pretraining_tensor_parallel_size

        if PositionEmbeddingType(self.position_embedding_type) == PositionEmbeddingType.alibi:
            raise NotImplementedError("currently DesyncResidual doesn't support alibi")

        self.reduce_pattern = (
            [{"attention": True, "mlp": True} for i in range(self.n_layer)]
            if reduce_pattern is None
            else reduce_pattern
        )
