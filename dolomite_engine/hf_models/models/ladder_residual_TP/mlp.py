import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Partial, Replicate

from ....distributed import dtensor_to_tensor, tensor_to_dtensor
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

    def forward(self, hidden_states):
        hidden_states = dtensor_to_tensor(
            tensor_to_dtensor(
                hidden_states,
                device_mesh=self.tp_mesh,
                current_placement=self.placement,
                desired_placement=Replicate(),
            )
        )
        hidden_states = F.linear(hidden_states, dtensor_to_tensor(self.c_fc.weight), dtensor_to_tensor(self.c_fc.bias))
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.linear(
            hidden_states, dtensor_to_tensor(self.c_proj.weight), dtensor_to_tensor(self.c_proj.bias)
        )
        hidden_states = dtensor_to_tensor(
            tensor_to_dtensor(
                hidden_states,
                device_mesh=self.tp_mesh,
                current_placement=Partial(),
                desired_placement=self.placement,
            )
        )
        hidden_states = self.dropout(hidden_states)
        return hidden_states
