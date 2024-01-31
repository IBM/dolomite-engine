from typing import List

import torch

from engine.constants import DatasetSplit


def get_num_samples_by_dataset(data_sampling_proportion: List[int], total_examples: int) -> List[int]:
    data_sampling_proportion = torch.tensor(data_sampling_proportion)
    num_samples_by_dataset = data_sampling_proportion / data_sampling_proportion.sum() * total_examples
    num_samples_by_dataset = num_samples_by_dataset.to(torch.long)
    num_samples_by_dataset[-1] = total_examples - num_samples_by_dataset[:-1].sum()
    return num_samples_by_dataset.tolist()


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
