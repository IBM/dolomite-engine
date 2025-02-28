import math

import torch
import torch.nn as nn

from .....utils import divide_if_divisible
from ....cache import HybridMambaAttentionDynamicCache
from ....enums import InitMethod
from ...activations import get_activation_function
from ...linear import ParameterizedLinear, ParameterizedConv1d
from ...mlp_blocks.mlp import _get_std_for_linear
from ...normalization import get_normalization_function
from .utils import _apply_mask_to_padding_states, _pad_tensor_by_size, _reshape_into_chunks, _segment_sum


class Mamba2Base(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        ssm_intermediate_size: int,
        ssm_num_heads: int,
        conv_kernel_size: int,
        time_step_limit: int,
        add_bias: bool,
        use_conv_bias: bool,
        ssm_activation_function: str,
        num_groups: int,
        chunk_size: int,
        layer_norm_epsilon: float,
        initializer_range: float,
        m_width: float,
        init_method: InitMethod,
        num_layers: int,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.num_heads = ssm_num_heads
        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = ssm_intermediate_size
        self.layer_idx = layer_idx
        self.use_conv_bias = use_conv_bias

        self.activation_string = ssm_activation_function
        self.activation = get_activation_function(self.activation_string)

        self.n_groups = num_groups
        self.head_dim = divide_if_divisible(ssm_intermediate_size, ssm_num_heads, "")
        self.chunk_size = chunk_size

        self.time_step_limit = time_step_limit
        # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
        assert ssm_intermediate_size == ssm_num_heads * head_dim
        std = _get_std_for_linear(initializer_range, init_method, m_width)

        # 1D convolutional layer
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = ParameterizedConv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            std=std,
        )

        # projection of the input hidden states
        self.in_proj = ParameterizedLinear(
            self.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads,
            bias=add_bias,
            std=std,
        )

        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        # Initialize log dt bias
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = get_normalization_function("silu_gated_rmsnorm", self.intermediate_size, eps=layer_norm_epsilon)
        self.D = nn.Parameter(torch.ones(self.num_heads))

        self.out_proj = ParameterizedLinear(
            self.intermediate_size, self.hidden_size, bias=add_bias, std=std / math.sqrt(2 * num_layers)
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        A = torch.arange(1, self.num_heads + 1)
        self.A_log.data = torch.log(A)
        nn.init.ones_(self.D)

    def forward(
        self,
        input_states: torch.Tensor,
        cache_params: HybridMambaAttentionDynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = input_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            input_states = (input_states * attention_mask[:, :, None]).to(dtype)

        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        input_states = _apply_mask_to_padding_states(input_states, attention_mask)
        projected_states = self.in_proj(input_states)
        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
        )

        # 2. Convolution sequence transformation
        if use_precomputed_states:
            cache_params.conv_states[self.layer_idx] = cache_params.conv_states[self.layer_idx].roll(
                shifts=-1, dims=-1
            )
            cache_params.conv_states[self.layer_idx][:, :, -1] = hidden_states_B_C[:, 0, :].to(
                cache_params.conv_states[self.layer_idx].device
            )

            # We need to guarantee that anything regarding the cache is on the same device
            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)

            hidden_states_B_C = torch.sum(conv_states * self.conv1d.weight.squeeze(1), dim=-1)
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Init cache
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = nn.functional.pad(
                    hidden_states_B_C_transposed, (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_states)

            hidden_states_B_C = self.activation(
                self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
            )

        hidden_states_B_C = _apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())  # [num_heads]
        if use_precomputed_states:
            # We need to guarantee that anything regarding the cache is on the same device
            cache_device = cache_params.ssm_states[self.layer_idx].device

            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [batch_size, num_heads] -> [batch_size, num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)


            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [num_heads, head_dim, state_size]
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)
            # [bsz, num_heads, head_dim, state_size]

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)

            # State calculation
            cache_params.ssm_states[self.layer_idx].copy_(cache_params.ssm_states[self.layer_idx] * dA + dBx)

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(
                device=C.device, dtype=C.dtype
            )  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(
                batch_size * self.num_heads, self.head_dim, self.ssm_state_size
            )  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            # dt: [batch_size, seq_len, num_heads]
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            # hidden_states: [batch_size, seq_len, num_heads, head_dim]
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            # B, C: [batch_size, seq_len, n_groups, ssm_state_size]
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            # B, C: [batch_size, seq_len, num_heads, ssm_state_size]
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
            D_residual = self.D[..., None] * _pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt
            # A: [batch_size, seq_len, num_heads]

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [
                _reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)
            ]

            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(_segment_sum(A))

            # Contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

            # Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            # 2. Compute the state for each intra-chunk
            # (right term of low-rank factorization of off-diagonal blocks; B terms)
            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
            # (middle term of factorization of off-diag blocks; A terms)
            if use_precomputed_states:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(_segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # 4. Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            # Init cache
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]

        return contextualized_states
