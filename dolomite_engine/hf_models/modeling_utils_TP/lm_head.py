import torch
import torch.nn.functional as F

from .embedding import Embedding_TP


class LMHead_TP(Embedding_TP):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight)
