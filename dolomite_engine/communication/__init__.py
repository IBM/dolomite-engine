from enum import Enum
from typing import Any

import torch.distributed
from torch.distributed import DeviceMesh, ProcessGroup, ReduceOp

from ..utils import ProcessGroupManager


class CommunicationBackend(Enum):
    torch_distributed = "torch_distributed"


class Communication:
    @staticmethod
    def broadcast_object(obj: Any, src: int, group: ProcessGroup) -> Any:
        if ProcessGroupManager.get_global_rank() != src:
            obj = None

        object_list = [obj]
        torch.distributed.broadcast_object_list(object_list, src=src, group=group)
        obj = object_list[0]

        return obj

    @staticmethod
    def all_reduce(
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        mesh: DeviceMesh | None = None,
        group: ProcessGroup | None = None,
        backend: CommunicationBackend | None = None,
    ) -> torch.Tensor:
        if mesh is not None:
            assert group is None
        elif group is not None:
            assert mesh is not None

        if backend == CommunicationBackend.torch_distributed:
            if mesh is not None:
                group = mesh.get_group()

            handle = torch.distributed.all_reduce(tensor, op=op, group=group, async_op=True)
        else:
            raise ValueError
