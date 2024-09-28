from ...config import CommonConfig


class GPTParallelConfig(CommonConfig):
    model_type = "gpt_parallel"

    def __init__(self, shared_layernorm: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)

        self.shared_layernorm = shared_layernorm
