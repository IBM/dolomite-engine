from typing import Any, Callable

from transformers import PretrainedConfig

from ..utils import BaseArgs
from .enums import AttentionHeadType, InitMethod, PositionEmbeddingType


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


class _MLPArgs(BaseArgs):
    mlp_block_type: str = "MLP"
    intermediate_size: int | None
    activation_function: str
    dropout: float
    initializer_range: float | None
    add_bias: bool
    m_width: float | None
    m_residual: float | None
    init_method: InitMethod

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_block_type == "MLP"


class _MoEArgs(_MLPArgs):
    mlp_block_type: str = "MoE"
    shared_intermediate_size: int | None
    num_experts: int
    num_experts_per_tok: int

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_block_type == "MoE"


class CommonConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50257,
        max_position_embeddings: int = 1024,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int | None = None,
        intermediate_size: int | None = None,
        activation_function: str = "gelu_pytorch_tanh",
        attention_head_type: str = "mqa",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        normalization_function: str = "layernorm",
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        attention_multiplier: float | None = None,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        attention_softmax_in_fp32: bool = True,
        add_bias: bool = True,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        rope_scaling: dict | None = None,
        m_emb: float | None = None,
        m_width: float | None = None,
        m_residual: float | None = None,
        init_method: str = "normal",
        upcast_logits_for_loss: bool = False,
        attention_blocks: list[str] = None,
        mlp_blocks: list[str] = None,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        router_aux_loss_coef: float = 0.001,
        shared_intermediate_size: int | None = None,
        use_aux_free_moe: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.activation_function = activation_function
        self.attention_head_type = attention_head_type
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.normalization_function = normalization_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.attention_multiplier = attention_multiplier
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.position_embedding_type = position_embedding_type
        self.add_bias = add_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.m_emb = m_emb
        self.m_width = m_width
        self.m_residual = m_residual
        self.init_method = init_method
        self.upcast_logits_for_loss = upcast_logits_for_loss

        if self.attention_multiplier is not None:
            assert self.scale_attn_weights

        # check if enums are valid
        init_method = InitMethod(init_method)
        attention_head_type = AttentionHeadType(attention_head_type)
        position_embedding_type = PositionEmbeddingType(position_embedding_type)

        # for compatibility with some features
        self.multi_query = attention_head_type == AttentionHeadType.mqa

        if attention_head_type == AttentionHeadType.mha:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.num_attention_heads

            assert (
                self.num_attention_heads == self.num_key_value_heads
            ), "MultiHeadAttention should have same number of heads for query, keys and values"
        elif attention_head_type == AttentionHeadType.mqa:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, "MultiQueryAttention should have 1 head for keys and values"
        elif attention_head_type == AttentionHeadType.gqa:
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert (
                self.num_attention_heads % self.num_key_value_heads == 0
            ), "GroupedQueryAttention should have more than 1 head for keys and values"

        self.attention_blocks = attention_blocks
        if self.attention_blocks is None:
            self.attention_blocks = [{"attention_block_type": "softmax_attention"} for _ in range(self.num_layers)]

        self.mlp_blocks = mlp_blocks
        self._set_mlp_blocks(
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            dropout=resid_pdrop,
            initializer_range=initializer_range,
            add_bias=add_bias,
            m_width=m_width,
            m_residual=m_residual,
            init_method=init_method,
            shared_intermediate_size=shared_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        assert len(self.attention_blocks) == self.num_layers
        assert len(self.mlp_blocks) == self.num_layers

        self.router_aux_loss_coef = router_aux_loss_coef

        self.use_aux_free_moe = use_aux_free_moe

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)

    @_hold_base_args(key="mlp_blocks")
    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs) -> None:
        return super().save_pretrained(save_directory, push_to_hub, **kwargs)

    @_hold_base_args(key="mlp_blocks")
    def to_json_string(self, use_diff: bool = True) -> str:
        return super().to_json_string(use_diff)

    def check_equal_for_all_and_get_value(self, key: str, key_block: str) -> Any:
        def _get(block, key):
            return block.get(key) if isinstance(block, dict) else getattr(block, key)

        blocks = getattr(self, key)
        expected_value = _get(blocks[0], key_block)

        assert all([_get(blocks[0], key_block) == expected_value for block in blocks])

        return expected_value

    def _set_mlp_blocks(
        self,
        intermediate_size: int,
        activation_function: str,
        dropout: float,
        initializer_range: float | None,
        add_bias: bool,
        m_width: float | None,
        m_residual: float | None,
        init_method: InitMethod,
        shared_intermediate_size: int | None,
        num_experts: int,
        num_experts_per_tok: int,
    ) -> None:
        if self.mlp_blocks is None:
            self.mlp_blocks = [{} for _ in range(self.num_layers)]

        mlp_blocks: list[_MLPArgs | _MoEArgs] = []
        for i in range(self.num_layers):
            mlp_block_type = self.mlp_blocks[i].get("mlp_block_type", "MLP")
            mlp_kwargs = dict(
                intermediate_size=self.mlp_blocks[i].get(
                    "intermediate_size", 4 * self.hidden_size if intermediate_size is None else intermediate_size
                ),
                activation_function=self.mlp_blocks[i].get("activation_function", activation_function),
                dropout=self.mlp_blocks[i].get("dropout", 0),
                initializer_range=self.mlp_blocks[i].get("initializer_range", initializer_range),
                add_bias=self.mlp_blocks[i].get("add_bias", add_bias),
                m_width=self.mlp_blocks[i].get("m_width", m_width),
                m_residual=self.mlp_blocks[i].get("m_residual", m_residual),
                init_method=self.mlp_blocks[i].get("init_method", init_method),
            )

            if mlp_block_type == "MLP":
                mlp_args = _MLPArgs(**mlp_kwargs)
            elif mlp_block_type == "MoE":
                mlp_args = _MoEArgs(
                    **mlp_kwargs,
                    shared_intermediate_size=self.mlp_blocks[i].get(
                        "shared_intermediate_size", shared_intermediate_size
                    ),
                    num_experts=self.mlp_blocks[i].get("num_experts", num_experts),
                    num_experts_per_tok=self.mlp_blocks[i].get("num_experts_per_tok", num_experts_per_tok),
                )

            mlp_blocks.append(mlp_args)

        self.mlp_blocks = mlp_blocks
