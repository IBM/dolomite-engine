import torch


def mask_probability(p: torch.Tensor, inference_method: dict | None) -> torch.Tensor:
    if inference_method is None:
        return p

    top_k = inference_method.get("top_k")
    threshold = inference_method.get("threshold")

    if threshold is not None:
        p = p.masked_fill(p < threshold, 0)
    elif top_k is not None:
        topk, indices = p.topk(top_k)
        p = torch.empty_like(p).scatter(-1, indices, topk)
    else:
        raise ValueError("unexpected inference_method")

    return p
