import torch
import torch.nn as nn

from ...modeling_utils import ParameterizedLinear


class EnsembleLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, tensor_parallel_size: int, std: float, bias: bool = True
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tensor_parallel_size = tensor_parallel_size
        self.std = std

        self.linears = nn.ModuleList(
            [ParameterizedLinear(in_features, out_features, bias=bias, std=std) for _ in range(tensor_parallel_size)]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = []
        for rank in range(self.tensor_parallel_size):
            if input.shape[1] == 1:
                input_ = input.squeeze(1)
            else:
                input_ = input[:, rank, ...]

            output_ = self.linears[rank](input_)
            output.append(output_)

        output = torch.stack(output, dim=1)

        return output
