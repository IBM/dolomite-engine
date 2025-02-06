from transformers import PretrainedConfig

from .enums import AttentionHeadType, InitMethod, PositionEmbeddingType


class CommonConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        vocab_size: int = 50257,
        max_position_embeddings: int = 1024,
        hidden_size: int = 768,
        num_layers: int = 12,
        n_head: int = 12,
        num_key_value_heads: int | None = None,
        n_inner: int | None = None,
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
        shared_n_inner: int | None = None,
        use_aux_free_moe: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_head = n_head
        self.num_key_value_heads = num_key_value_heads
        self.n_inner = 4 * hidden_size if n_inner is None else n_inner
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
                self.num_key_value_heads = self.n_head

            assert (
                self.n_head == self.num_key_value_heads
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
                self.n_head % self.num_key_value_heads == 0
            ), "GroupedQueryAttention should have more than 1 head for keys and values"

        self.attention_blocks = attention_blocks
        if self.attention_blocks is None:
            self.attention_blocks = [{"attention_block_type": "softmax_attention"} for _ in range(self.num_layers)]

        self.mlp_blocks = mlp_blocks
        if self.mlp_blocks is None:
            self.mlp_blocks = [{"mlp_block_type": "MLP"} for _ in range(self.num_layers)]

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.shared_n_inner = shared_n_inner

        self.router_aux_loss_coef = router_aux_loss_coef

        self.use_aux_free_moe = use_aux_free_moe

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
