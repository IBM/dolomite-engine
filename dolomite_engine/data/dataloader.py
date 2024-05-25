from typing import Any, Callable, Iterable, List, Tuple

import torch
import torch.distributed
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader, Dataset, Sampler

from ..utils import get_global_rank


class ResumableDataLoader(DataLoader):
    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict(), "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset.load_state_dict(state_dict.get("dataset"))
        self.sampler.load_state_dict(state_dict.get("sampler"))


class DispatchingDataLoader(ResumableDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[List] | Iterable[List] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[List], Any] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        source_ranks_broadcast_ranks_broadcast_groups: List[Tuple[int, List[int], ProcessGroup]] = None,
        keys: List[str] = ["input_ids", "attention_mask", "labels"],
    ) -> None:
        self.broadcast_world_size = len(source_ranks_broadcast_ranks_broadcast_groups[0][1])
        self.all_source_ranks_and_broadcast_groups = source_ranks_broadcast_ranks_broadcast_groups

        global_rank = get_global_rank()

        self.is_source = False
        for src, _, _ in self.all_source_ranks_and_broadcast_groups:
            if src == global_rank:
                self.is_source = True
                break

        self.local_rank_in_broadcast_group = None
        for _, broadcast_ranks, _ in self.all_source_ranks_and_broadcast_groups:
            if global_rank in broadcast_ranks:
                self.local_rank_in_broadcast_group = broadcast_ranks.index(global_rank)
                break
        assert self.local_rank_in_broadcast_group is not None

        super().__init__(
            dataset=dataset,
            batch_size=batch_size * self.broadcast_world_size if batch_sampler is None else 1,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        _length = torch.tensor(
            [super().__len__() if self.is_source else 0], dtype=torch.long, device=torch.cuda.current_device()
        )
        self._broadcast(_length)
        self._length = _length.item()

        self.keys = keys

    def _broadcast(self, item: torch.Tensor, is_tensor: bool = True) -> None:
        for src, _, grp in self.all_source_ranks_and_broadcast_groups:
            if is_tensor:
                torch.distributed.broadcast(item, src=src, group=grp)
            else:
                torch.distributed.broadcast_object_list(item, src=src, group=grp)

    def __iter__(self):
        iterator = super().__iter__() if self.is_source else range(self._length)

        for batch in iterator:
            # if using dynamic shapes at every batch or when batch buffer is None during static batch, we need to get shape
            # send/recv tensor shapes
            batch_shape = [batch[self.keys[0]].shape if self.is_source else None]
            self._broadcast(batch_shape, is_tensor=False)
            batch_shape = batch_shape[0]

            if self.is_source:
                for key in self.keys:
                    batch[key] = batch[key].to(torch.cuda.current_device())
            else:
                batch = {
                    key: torch.empty(batch_shape, dtype=torch.long, device=torch.cuda.current_device())
                    for key in self.keys
                }

            for key in batch:
                # send/recv batch
                self._broadcast(batch[key])

                # slice batch
                local_batch_size = batch[key].shape[0] // self.broadcast_world_size
                batch[key] = batch[key][
                    self.local_rank_in_broadcast_group
                    * local_batch_size : (self.local_rank_in_broadcast_group + 1)
                    * local_batch_size
                ]

            yield batch

    def __len__(self) -> int:
        return self._length
