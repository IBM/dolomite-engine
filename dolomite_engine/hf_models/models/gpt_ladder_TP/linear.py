import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Shard

from ...modeling_utils_TP import ColumnParallelLinear, dtensor_to_tensor, tensor_to_dtensor


class LadderColumnParallelLinear(ColumnParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        default_stream = torch.cuda.default_stream(torch.cuda.current_device())

        with torch.cuda.stream(default_stream):
            input = tensor_to_dtensor(input, current_placement=self.input_placement)

        input = F.linear(input, self.weight, self.bias)
        input = dtensor_to_tensor(input, desired_placement=Shard(-1))
        return input
