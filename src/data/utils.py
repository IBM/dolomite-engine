from typing import List

import torch


def get_num_samples_by_dataset(data_sampling_proportion: List[int], total_samples: int) -> List[int]:
    data_sampling_proportion = torch.tensor(data_sampling_proportion)
    num_samples_by_dataset = data_sampling_proportion / data_sampling_proportion.sum() * total_samples
    num_samples_by_dataset = num_samples_by_dataset.to(torch.long)
    num_samples_by_dataset[-1] = total_samples - num_samples_by_dataset[:-1].sum()
    return num_samples_by_dataset.tolist()
