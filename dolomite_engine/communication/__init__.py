from typing import Any

import torch.distributed
import torch.distributed._functional_collectives as funcol
from torch.distributed import DeviceMesh, ProcessGroup, ReduceOp

from ..utils import ProcessGroupManager
from .backend import CommunicationBackend


_REDUCE_OP_MAP = {ReduceOp.SUM: "sum"}


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
        op: ReduceOp | str = ReduceOp.SUM,
        mesh: DeviceMesh | None = None,
        group: ProcessGroup | None = None,
        backend: CommunicationBackend | None = None,
    ) -> torch.Tensor:
        if backend == CommunicationBackend.torch_distributed:
            if mesh is not None:
                assert group is None
                group = mesh.get_group()
            elif group is not None:
                assert mesh is not None

            work = torch.distributed.all_reduce(tensor, op=op, group=group, async_op=True)
            torch._C._distributed_c10d._register_work(tensor, work)

            tensor = funcol.AsyncCollectiveTensor(tensor)
        elif backend == CommunicationBackend.torch_functional:
            if not isinstance(op, str):
                op = _REDUCE_OP_MAP[op]

            tensor = funcol.all_reduce(tensor, reduceOp=op, group=mesh or group)
        elif backend == CommunicationBackend.mscclpp:
            raise ValueError

        return tensor
