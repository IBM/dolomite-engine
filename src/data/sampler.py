import math
from argparse import Namespace
from typing import Iterator, List

import torch
from torch.utils.data import DistributedSampler

from src.data.dataset import ConcatenatedDatasets


class ConcatenatedDataSampler(DistributedSampler):
    def __init__(
        self,
        args: Namespace,
        dataset: ConcatenatedDatasets,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.data_sampling_proportion: List[int] = args.data_sampling_proportion
        self.num_datasets = dataset.num_datasets

    def get_indices_in_data_subset(self, num_samples: int, subset_size: int, seed: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(seed)

        if num_samples < subset_size:
            sampler = torch.randperm(num_samples, generator=g)
        else:
            num_concats = num_samples // subset_size
            padding = num_samples - num_concats * subset_size
            sampler = list(range(subset_size)) * num_concats
            sampler = torch.tensor(sampler)

            if padding > 0:
                padding_samples = torch.randperm(subset_size, generator=g)
                padding_samples = padding_samples[:padding]
                sampler = torch.cat([sampler, padding_samples])

        return sampler

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            num_data_samples = torch.tensor(self.data_sampling_proportion)
            num_data_samples = num_data_samples / num_data_samples.sum() * len(self.dataset)
            num_data_samples = num_data_samples.to(torch.long)
            num_data_samples[-1] = len(self.dataset) - num_data_samples[:-1].sum()

            data_samples = []
            for i in range(self.num_datasets):
                start_index = self.dataset.start_indices[i]
                sampler = self.get_indices_in_data_subset(
                    num_data_samples[i].item(), len(self.dataset.datasets[i]), self.seed + (self.epoch + 1) * (i + 1)
                )
                sampler += start_index

                data_samples.extend(sampler.tolist())

            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(data_samples), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        for i in indices:
            if self.shuffle:
                yield data_samples[i]
            else:
                yield i
