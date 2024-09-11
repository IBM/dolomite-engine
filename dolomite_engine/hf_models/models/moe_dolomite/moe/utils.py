import torch
from torch.nn import functional as F


def compute_switchloss(logits: torch.Tensor, probs: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
    logits = logits.view(-1, logits.size(-1))
    probs = probs.view(-1, probs.size(-1))

    num_experts = logits.size(1)
    acc_probs = probs.sum(0)
    freq = torch.bincount(topk_idxs.flatten(), minlength=num_experts).to(dtype=logits.dtype)

    switchloss = num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
    zloss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

    loss = switchloss + 0.1 * zloss

    return loss
