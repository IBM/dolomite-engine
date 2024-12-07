import torch
import torch.nn.functional as F

from ....distributed import dtensor_to_tensor
from ....utils import ProcessGroupManager
from ...modeling_utils_TP import get_module_placements
from ..gpt_dolomite_TP.mlp import MLP_TP
from ..ladder_residual import LadderResidualConfig


class LadderMLP_TP(MLP_TP):
    def __init__(
        self, config: LadderResidualConfig, use_padding_free_transformer: bool = False, sequence_parallel: bool = False
    ) -> None:
        super().__init__(config, use_padding_free_transformer, sequence_parallel)
        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)
        assert config.resid_pdrop == 0

    def forward(self, current_attention_out: torch.Tensor, current_mlp_out: torch.Tensor) -> tuple[torch.Tensor]:
        current_mlp_out = F.linear(
            current_mlp_out, dtensor_to_tensor(self.c_fc.weight), dtensor_to_tensor(self.c_fc.bias)
        )
        current_mlp_out = self.act(current_mlp_out)
        current_mlp_out = F.linear(
            current_mlp_out, dtensor_to_tensor(self.c_proj.weight), dtensor_to_tensor(self.c_proj.bias)
        )
        return current_attention_out, current_mlp_out
