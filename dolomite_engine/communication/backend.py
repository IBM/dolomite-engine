from enum import Enum


class CommunicationBackend(Enum):
    torch_distributed = "torch_distributed"
    torch_functional = "torch_functional"
