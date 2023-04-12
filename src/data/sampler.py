import math
from argparse import Namespace
from typing import Iterator, List

import torch
from torch.utils.data import DistributedSampler

from src.constants import DatasetSplit, Mode
from src.data.dataset import ConcatenatedDatasets
from src.data.utils import get_num_samples_by_dataset
from src.utils.distributed import get_world_size
from src.utils.logging import print_rank_0


class ConcatenatedDataSampler(DistributedSampler):
    """Data sampler used for training on multiple datasets according to the specified sampling proportions"""

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

        self.dataset: ConcatenatedDatasets

        self.num_examples_in_each_dataset = self.dataset.get_num_examples_in_each_dataset()
        self.num_datasets = dataset.get_num_datasets()

        if args.ignore_sampling_proportion_for_validation and self.dataset.split == DatasetSplit.val:
            self.num_samples_by_dataset = self.num_examples_in_each_dataset
        else:
            self.num_samples_by_dataset = get_num_samples_by_dataset(args.data_sampling_proportion, len(dataset))

        self.print_sampler_stats(
            args.batch_size_per_gpu if self.dataset.mode == Mode.training else args.batch_size, args.num_training_steps
        )

    def get_indices_in_data_subset(self, num_samples_in_subset: int, subset_size: int, seed: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(seed)

        if num_samples_in_subset < subset_size:
            sampler = torch.randperm(num_samples_in_subset, generator=g)
        else:
            num_concats = num_samples_in_subset // subset_size
            padding = num_samples_in_subset - num_concats * subset_size
            sampler = list(range(subset_size)) * num_concats
            sampler = torch.tensor(sampler)

            if padding > 0:
                padding_samples = torch.randperm(subset_size, generator=g)
                padding_samples = padding_samples[:padding]
                sampler = torch.cat([sampler, padding_samples])

        return sampler

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            data_samples = []

            for i in range(self.num_datasets):
                sampler = self.get_indices_in_data_subset(
                    self.num_samples_by_dataset[i],
                    self.num_examples_in_each_dataset[i],
                    self.seed + (self.epoch + 1) * (i + 1),
                )
                sampler += self.dataset.start_indices[i]

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

    def print_sampler_stats(self, batch_size_per_gpu: int, num_training_steps: int) -> None:
        """prints the statistics of the program"""

        if self.dataset.mode == Mode.training and self.dataset.split == DatasetSplit.train:
            num_steps = num_training_steps
        elif self.dataset.mode == Mode.inference or self.dataset.split != DatasetSplit.train:
            examples_per_step = batch_size_per_gpu * get_world_size()

            num_steps = len(self.dataset) // examples_per_step
            if len(self.dataset) % examples_per_step != 0:
                num_steps = (len(self.dataset) // examples_per_step) + 1

        print_rank_0(f"{'*' * 25} {self.dataset.split.value} {'*' * 25}")

        print_rank_0(f"total samples in 1 epoch of the dataset mixture = {len(self.dataset)}")
        print_rank_0(
            f"total epochs for the dataset mixture = {num_steps * batch_size_per_gpu * get_world_size() / len(self.dataset)}"
        )

        for i, dataset in enumerate(self.dataset.datasets):
            print_rank_0(
                f"\nnumber of samples of {dataset.__class__.__name__} in 1 epoch of the entire dataset = {self.num_samples_by_dataset[i]}"
            )
            print_rank_0(
                f"number of epochs of {dataset.__class__.__name__} in 1 epoch of the entire dataset = {self.num_samples_by_dataset[i] / len(dataset)}"
            )

        print_rank_0("*" * 50)
