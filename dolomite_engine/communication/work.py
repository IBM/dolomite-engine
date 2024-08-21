from datetime import timedelta

from torch.distributed import Work

from .backend import CommunicationBackend


class DolomiteWork(Work):
    def __init__(self, backend: CommunicationBackend) -> None:
        super().__init__()
        self.backend = backend

    def wait(self, timeout: timedelta = ...) -> bool:
        if (
            self.backend == CommunicationBackend.torch_distributed
            or self.backend == CommunicationBackend.torch_functional
        ):
            output = super().wait(timeout)

        return output
