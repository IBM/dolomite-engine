import torch
import torch.nn as nn


class EnsembleLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, tensor_parallel_size: int, std: float, bias: bool = True
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tensor_parallel_size = tensor_parallel_size
        self.std = std

        self.weight = nn.Parameter(torch.empty(tensor_parallel_size, in_features, out_features))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(tensor_parallel_size, out_features))

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input -> (batch_size, sequence_length, TP, in_features)
        input = input.unsqueeze(-2)
        # input -> (batch_size, sequence_length, TP, 1, in_features)
        weight = self.weight.unsqueeze(0).unsqueeze(0)
        # weight -> (1, 1, TP, in_features, out_features)

        input = input @ weight
        # input -> (batch_size, sequence_length, TP, 1, out_features)

        if self.bias is not None:
            bias = self.bias.unsqueeze(1).unsqueeze(0).unsqueeze(0)
            # bias -> (1, 1, TP, 1, out_features)
            input = input + bias

        input = input.squeeze(-2)
        # input -> (batch_size, sequence_length, TP, out_features)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if self.bias is not None:
            self.bias.zero_()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, bias={self.bias is not None}"
        )
