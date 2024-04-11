from typing import List

import torch
import torch.distributed


class ProcessGroupManager:
    def __init__(self, ranks: List[int] = None) -> None:
        self.ranks = list(range(torch.distributed.get_world_size())) if ranks is None else ranks
        self.process_group: torch.distributed.ProcessGroup = torch.distributed.new_group(self.ranks)
        self.world_size = torch.distributed.get_world_size(self.process_group)
        self.rank = torch.distributed.get_rank(self.process_group)

    def get_process_group(self) -> torch.distributed.ProcessGroup:
        return self.process_group

    def get_ranks(self) -> List[int]:
        return self.ranks

    def get_world_size(self) -> int:
        return self.world_size

    def get_first_rank(self) -> int:
        return self.ranks[0]

    def get_rank(self) -> int:
        return self.rank
