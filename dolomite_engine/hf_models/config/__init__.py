from copy import deepcopy
from typing import Any, Callable

from transformers import PretrainedConfig

from ...utils import BaseArgs
from ..enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from .mlp import _MLPArgs, _MoEArgs
from .sequence_mixer import _SoftmaxAttentionArgs


def _hold_base_args(key: str) -> Callable:
    def _holded_function(function: Callable) -> Callable:
        def _run(self, *args, **kwargs):
            value: list[BaseArgs] = getattr(self, key)
            setattr(self, key, [i.to_dict() if isinstance(i, BaseArgs) else i for i in value])
            output = function(self, *args, **kwargs)
            setattr(self, key, value)
            return output

        return _run

    return _holded_function


def _update_with_key_value(block: dict, kwargs: dict, key: str) -> None:
    if key in block:
        kwargs[key] = block.pop(key)


_NAKED_DISALLOWED_ARGS = [
    "activation_function",
    "attn_pdrop",
    "embd_pdrop",
    "resid_pdrop",
    "intermediate_size",
    "shared_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "add_bias",
    "attention_blocks",
    "num_key_value_heads",
    "attention_head_type",
    "attention_multiplier",
]


class CommonConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50304,
        max_position_embeddings: int = 1024,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        embedding_dropout: float = 0,
        normalization_function: str = "layernorm",
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        rope_scaling: dict | None = None,
        m_emb: float | None = None,
        m_width: float | None = None,
        m_residual: float | None = None,
        init_method: str = "normal",
        upcast_logits_for_loss: bool = False,
        sequence_mixer_blocks: list[dict] = None,
        mlp_blocks: list[dict] = None,
        router_aux_loss_coef: float = 0.001,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout = embedding_dropout
        self.normalization_function = normalization_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.m_emb = m_emb
        self.m_width = m_width
        self.m_residual = m_residual
        self.init_method = init_method
        self.upcast_logits_for_loss = upcast_logits_for_loss

        # check if enums are valid
        init_method = InitMethod(init_method)
        position_embedding_type = PositionEmbeddingType(position_embedding_type)

        self.sequence_mixer_blocks = sequence_mixer_blocks
        self._set_sequence_mixer_blocks()
        assert len(self.sequence_mixer_blocks) == self.num_layers

        self.mlp_blocks = mlp_blocks
        self._set_mlp_blocks()
        assert len(self.mlp_blocks) == self.num_layers

        self.router_aux_loss_coef = router_aux_loss_coef

        for i in _NAKED_DISALLOWED_ARGS:
            assert i not in kwargs, f"i found ({i})"

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)

    @_hold_base_args(key="sequence_mixer_blocks")
    @_hold_base_args(key="mlp_blocks")
    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs) -> None:
        return super().save_pretrained(save_directory, push_to_hub, **kwargs)

    @_hold_base_args(key="sequence_mixer_blocks")
    @_hold_base_args(key="mlp_blocks")
    def to_json_string(self, use_diff: bool = True) -> str:
        return super().to_json_string(use_diff)

    def check_equal_for_all_and_get_value(self, key: str, key_block: str, expected_value: Any | None = None) -> Any:
        def _get(block, key):
            return block.get(key) if isinstance(block, dict) else getattr(block, key)

        blocks = getattr(self, key)
        value = _get(blocks[0], key_block)

        if expected_value is not None:
            assert value == expected_value, f"{value} {expected_value}"

        assert all([_get(block, key_block) == value for block in blocks])

        return value

    def _set_sequence_mixer_blocks(self) -> None:
        if self.sequence_mixer_blocks is None:
            self.sequence_mixer_blocks = [{} for _ in range(self.num_layers)]

        sequence_mixer_blocks: list[_SoftmaxAttentionArgs] = []
        for i in range(self.num_layers):
            sequence_mixer_block = deepcopy(self.sequence_mixer_blocks[i])
            sequence_mixer_block_type = sequence_mixer_block.get("mlp_block_type", "softmax_attention")

            attention_head_type = AttentionHeadType(sequence_mixer_block.get("attention_head_type", "mqa"))
            num_key_value_heads = sequence_mixer_block.get("num_key_value_heads", None)

            if attention_head_type == AttentionHeadType.mha:
                if num_key_value_heads is None:
                    num_key_value_heads = self.num_attention_heads

                assert (
                    self.num_attention_heads == num_key_value_heads
                ), "MultiHeadAttention should have same number of heads for query, keys and values"
            elif attention_head_type == AttentionHeadType.mqa:
                if num_key_value_heads is None:
                    num_key_value_heads = 1

                assert num_key_value_heads == 1, "MultiQueryAttention should have 1 head for keys and values"
            elif attention_head_type == AttentionHeadType.gqa:
                assert (
                    num_key_value_heads is not None
                ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

                assert (
                    self.num_attention_heads % num_key_value_heads == 0
                ), "GroupedQueryAttention should have more than 1 head for keys and values"

            sequence_mixer_kwargs = dict(
                num_key_value_heads=num_key_value_heads,
                attention_head_type=attention_head_type,
                softmax_dropout=sequence_mixer_block.get("softmax_dropout", 0),
                dropout=sequence_mixer_block.get("dropout", 0),
                add_bias=sequence_mixer_block.get("add_bias", True),
                attention_multiplier=sequence_mixer_block.get("attention_multiplier", None),
            )

            if sequence_mixer_block_type == "softmax_attention":
                sequence_mixer_args = _SoftmaxAttentionArgs(**sequence_mixer_kwargs)
            else:
                raise ValueError(f"unexpected sequence_mixer_block_type ({sequence_mixer_block_type})")

            sequence_mixer_blocks.append(sequence_mixer_args)

        self.sequence_mixer_blocks = sequence_mixer_blocks

    def _set_mlp_blocks(self) -> None:
        if self.mlp_blocks is None:
            self.mlp_blocks = [{} for _ in range(self.num_layers)]

        mlp_blocks: list[_MLPArgs | _MoEArgs] = []
        for i in range(self.num_layers):
            mlp_block = deepcopy(self.mlp_blocks[i])
            mlp_block_type = mlp_block.pop("mlp_block_type", "MLP")

            mlp_kwargs = {"intermediate_size": mlp_block.pop("intermediate_size", 4 * self.hidden_size)}

            _update_with_key_value(mlp_block, mlp_kwargs, "activation_function")
            _update_with_key_value(mlp_block, mlp_kwargs, "dropout")
            _update_with_key_value(mlp_block, mlp_kwargs, "add_bias")

            if mlp_block_type == "MLP":
                mlp_class = _MLPArgs
            elif mlp_block_type == "MoE":
                _update_with_key_value(mlp_block, mlp_kwargs, "shared_intermediate_size")
                _update_with_key_value(mlp_block, mlp_kwargs, "num_experts")
                _update_with_key_value(mlp_block, mlp_kwargs, "num_experts_per_tok")

                mlp_class = _MoEArgs
            else:
                raise ValueError(f"unexpected mlp_block_type ({mlp_block_type})")

            assert len(mlp_block) == 0, f"leftover keys in the mlp_block ({mlp_block}) at position {i}"
            mlp_blocks.append(mlp_class(**mlp_kwargs))

        self.mlp_blocks = mlp_blocks
