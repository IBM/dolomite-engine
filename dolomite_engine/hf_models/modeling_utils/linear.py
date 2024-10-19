import torch
import torch.nn as nn


class ParameterizedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        self.std = std
        super().__init__(in_features, out_features, bias, device, dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.zero_()


class ParameterizedTransposedLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        self.std = std

        if bias:
            raise NotImplementedError("bias is not supported with TransposedLinear yet")

        # pass in_features as out_features and vice-versa
        super().__init__(out_features, in_features, bias, device, dtype)

        # invert them now to print the module correctly
        self.in_features, self.out_features = self.out_features, self.in_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ self.weight
