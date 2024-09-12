from os import PathLike

from ...config import CommonConfig


class GPTCrossLayerConfig(CommonConfig):
    model_type = "gpt_crosslayer"

    def __init__(
        self,
        attention_head_type: str = "gqa",
        num_key_value_heads: int = 12,
        sharing_pattern: list[int] | None = None,
        joint_residual_stream: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(attention_head_type=attention_head_type, num_key_value_heads=num_key_value_heads, **kwargs)

        if sharing_pattern is None:
            self.sharing_pattern = list(range(self.n_layer))
        else:
            assert all(
                [sharing_pattern[i] == i for i in set(sharing_pattern)]
            ), "a filled sharing pattern doesn't have a parent layer"

            for i in range(len(sharing_pattern) - 1):
                assert sharing_pattern[i] <= sharing_pattern[i + 1]

            for i in range(len(sharing_pattern)):
                assert sharing_pattern[i] >= 0
                assert sharing_pattern[i] < self.n_layer

            self.sharing_pattern = sharing_pattern

        self.joint_residual_stream = joint_residual_stream

        assert self.init_method == "normal"

    def save_pretrained(self, save_directory: str | PathLike, push_to_hub: bool = False, **kwargs):
        # we don't want to exporet attention_head_type in this config, not sure if there is a better way to do this
        attention_head_type = self.attention_head_type
        del self.attention_head_type

        output = super().save_pretrained(save_directory, push_to_hub, **kwargs)

        self.attention_head_type = attention_head_type

        return output
