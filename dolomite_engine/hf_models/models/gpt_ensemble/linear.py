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

        self.weight = nn.Parameter(torch.empty(self.tensor_parallel_size, self.in_features, self.out_features))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(self.tensor_parallel_size, self.out_features))

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_length, _ = input.shape

        if input.shape[1] != self.tensor_parallel_size:
            assert input.shape[1] == 1
            input = input.expand(-1, self.tensor_parallel_size, -1, -1)

        # input -> (batch_size, TP, sequence_length, in_features)
        input = input.transpose(0, 1)
        # input -> (TP, batch_size, sequence_length, in_features)
        input = input.reshape(self.tensor_parallel_size, batch_size * sequence_length, -1)
        # input -> (TP, batch_size * sequence_length, in_features)

        if self.bias is None:
            input = torch.bmm(input, self.weight)
        else:
            input = torch.baddbmm(self.bias.unsqueeze(1), input, self.weight, alpha=1, beta=1)

        # input -> (TP, batch_size * sequence_length, out_features)
        input = input.reshape(self.tensor_parallel_size, batch_size, sequence_length, -1)
        # (TP, batch_size, sequence_length, out_features)
        input = input.transpose(0, 1).contiguous()

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
