import math

from ...config import CommonConfig


class Mamba2DolomiteConfig(CommonConfig):
    model_type = "mamba2_dolomite"

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
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
        layer_map: list[str] | None = None,
        conv_kernel_size: int = 4,
        n_groups: int = 8,
        use_conv_bias: bool = True,
        time_step_rank: str = "auto",
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 1e-4,
        time_step_limit: float = (0.0, float("inf")),
        chunk_size: int = 256,
        ssm_activation_function: str = "silu",
        ssm_state_size: int = 128,
        ssm_head_dim: int = 64,
        ssm_num_heads: int = 128,
        ssm_intermediate_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_size,
            n_positions,
            n_embd,
            n_layer,
            n_head,
            num_key_value_heads,
            n_inner,
            activation_function,
            attention_head_type,
            resid_pdrop,
            embd_pdrop,
            attn_pdrop,
            normalization_function,
            layer_norm_epsilon,
            initializer_range,
            scale_attn_weights,
            attention_multiplier,
            use_cache,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            attention_softmax_in_fp32,
            add_bias,
            position_embedding_type,
            rope_theta,
            rope_scaling,
            m_emb,
            m_width,
            m_residual,
            init_method,
            upcast_logits_for_loss,
            **kwargs,
        )

        self.layer_map = ["mamba2"] * self.n_layer if layer_map is None else layer_map
        self.conv_kernel_size = conv_kernel_size
        self.n_groups = n_groups
        self.use_conv_bias = use_conv_bias

        self.time_step_limit = time_step_limit
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor

        self.chunk_size = chunk_size

        self.ssm_activation_function = ssm_activation_function
        self.ssm_state_size = ssm_state_size
        self.ssm_head_dim = ssm_head_dim
        self.ssm_num_heads = ssm_num_heads
        self.ssm_intermediate_size = 2 * self.n_embd if ssm_intermediate_size is None else ssm_intermediate_size
