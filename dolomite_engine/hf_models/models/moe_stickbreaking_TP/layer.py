import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard
from torch.nn import functional as F
from transformers import DynamicCache

from ....distributed import dtensor_to_tensor, tensor_to_dtensor
from ....utils import ProcessGroupManager, divide_if_divisible
from ...modeling_utils_TP import DTensorModule, get_attention_module_TP, get_normalization_function_TP
from ...modeling_utils_TP.attention.base import Attention_TP, _BaseAttention_TP
from ...modeling_utils_TP.attention.padding_free import PaddingFreeAttention_TP
from ...modeling_utils_TP.TP import get_module_placements
from ..moe_dolomite_TP.moe_TP import ScatterMoE_TP
from ..moe_stickbreaking import MoEStickBreakingConfig
from ..moe_stickbreaking.layer import MoEStickBreakingBlock, PaddingFreeSBAttention


class _GroupNorm(nn.GroupNorm):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        # use_padding_free_transformer: bool = False,
        # sequence_parallel: bool = False,
    ) -> None:
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
        self.weight = nn.Parameter(
            tensor_to_dtensor(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                current_placement=Shard(0),
            )
        )
        self.bias = nn.Parameter(
            tensor_to_dtensor(
                self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Shard(0)
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = dtensor_to_tensor(self.weight)
        bias = dtensor_to_tensor(self.bias)
        input = F.group_norm(input, self.num_groups, weight, bias, self.eps)
        return input


class PaddingFreeSBAttention_TP(_BaseAttention_TP, PaddingFreeSBAttention):
    def __init__(
        self,
        config: MoEStickBreakingConfig,
        causal: bool,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        _BaseAttention_TP.__init__(
            self,
            config=config,
            causal=causal,
            layer_idx=layer_idx,
            use_padding_free_transformer=True,
            sequence_parallel=sequence_parallel,
        )

        self.sb_remainder = config.sb_remainder
        if self.sb_remainder:
            self.head_bias = torch.nn.Parameter(
                tensor_to_dtensor(
                    torch.zeros(self.hidden_size // self.head_dim, self.head_dim),
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    current_placement=Shard(0),
                )
            )

        self.norm = _GroupNorm(self.num_heads, self.hidden_size)

    def _get_head_bias(self):
        return dtensor_to_tensor(self.head_bias)[:, None, :]

    def _prepare_qkv_for_forward_mqa(
        self, query_key_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = query_key_value

        total_q = query.shape[0]

        query = query.view(total_q, self.num_heads, -1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value


class MoEStickBreakingBlock_TP(MoEStickBreakingBlock):
    def __init__(
        self,
        config: MoEStickBreakingConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        moe_implementation: str,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        assert use_padding_free_transformer
        # self.attn = PaddingFreeAttention_TP(
        self.attn = PaddingFreeSBAttention_TP(
            config,
            True,
            layer_idx=layer_idx,
            sequence_parallel=sequence_parallel,
        )

        self.ln_2 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        assert moe_implementation == "scattermoe", "TP for MoE is only implemented with scattermoe"
        self.moe = ScatterMoE_TP(
            config,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            layer_idx=layer_idx,
        )
