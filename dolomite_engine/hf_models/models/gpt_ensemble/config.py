from ...config import CommonConfig
from ...enums import PositionEmbeddingType


class GPTEnsembleConfig(CommonConfig):
    model_type = "gpt_ensemble"

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        num_key_value_heads: int = None,
        n_inner: int = None,
        activation_function: str = "gelu_pytorch_tanh",
        attention_head_type: str = "mqa",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        normalization_function: str = "layernorm",
        layer_norm_epsilon: float = 0.00001,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        attention_multiplier: float = None,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        attention_softmax_in_fp32: bool = True,
        add_bias: bool = True,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        rope_scaling: dict = None,
        m_emb: float = None,
        m_width: float = None,
        m_residual: float = None,
        init_method: str = "normal",
        upcast_logits_for_loss: bool = False,
        pretraining_tensor_parallel_size: int = 1,
        reduce_pattern: dict | None = None,
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
            attention_head_type=attention_head_type,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            normalization_function=normalization_function,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            attention_multiplier=attention_multiplier,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            attention_softmax_in_fp32=attention_softmax_in_fp32,
            add_bias=add_bias,
            position_embedding_type=position_embedding_type,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            m_emb=m_emb,
            m_width=m_width,
            m_residual=m_residual,
            init_method=init_method,
            upcast_logits_for_loss=upcast_logits_for_loss,
            **kwargs,
        )

        self.pretraining_tensor_parallel_size = pretraining_tensor_parallel_size

        if position_embedding_type == PositionEmbeddingType.alibi:
            raise NotImplementedError("currently GPTEnsemble doesn't support alibi")

        self.reduce_pattern = (
            [{"attention": True, "mlp": True} for i in range(n_layer)] if reduce_pattern is None else reduce_pattern
        )
