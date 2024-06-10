from os import PathLike
from typing import List

from ...config import CommonConfig


class GPTCrossLayerConfig(CommonConfig):
    model_type = "gpt_crosslayer"

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        num_key_value_heads: int = 12,
        n_inner: int = None,
        activation_function: str = "gelu_pytorch_tanh",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        normalization_function: str = "layernorm",
        layer_norm_epsilon: float = 0.00001,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        add_bias: bool = True,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        rope_scaling: dict = None,
        sharing_pattern: List[int] = None,
        joint_residual_stream: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            num_key_value_heads=num_key_value_heads,
            n_inner=n_inner,
            activation_function=activation_function,
            attention_head_type="gqa",
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            normalization_function=normalization_function,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            add_bias=add_bias,
            position_embedding_type=position_embedding_type,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            **kwargs,
        )

        if sharing_pattern is None:
            self.sharing_pattern = list(range(n_layer))
        else:
            assert all(
                [sharing_pattern[i] == i for i in set(sharing_pattern)]
            ), "a filled sharing pattern doesn't have a parent layer"

            for i in range(len(sharing_pattern) - 1):
                assert sharing_pattern[i] <= sharing_pattern[i + 1]

            for i in range(len(sharing_pattern)):
                assert sharing_pattern[i] >= 0
                assert sharing_pattern[i] < n_layer

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
