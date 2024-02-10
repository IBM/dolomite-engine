import torch

from ..enums import DatasetSplit


def train_val_test_split(
    data: list, split: DatasetSplit, seed: int, val_samples: int, test_samples: int
) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(data), generator=g)

    if split == DatasetSplit.train:
        indices = indices[: -(val_samples + test_samples)]
    elif split == DatasetSplit.val:
        indices = indices[-(val_samples + test_samples) : -test_samples]
    elif split == DatasetSplit.test:
        indices = indices[-test_samples:]

    split_data = [data[i.item()] for i in indices]

    return split_data
