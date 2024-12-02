from enum import Enum

from .parallel import ProcessGroupManager


_MICRO_BATCH_SIZE: int | None = None
_GRADIENT_ACCUMULATION_STEPS: int | None = None
_SEQUENCE_LENGTH: int | None = None


class StepTrackerMethod(Enum):
    samples = "samples"
    tokens = "tokens"


class StepTracker:
    def __init__(self, micro_batch_size: int, gradient_accumulation_steps: int, sequence_length: int = None) -> None:
        global _MICRO_BATCH_SIZE, _GRADIENT_ACCUMULATION_STEPS, _SEQUENCE_LENGTH

        _MICRO_BATCH_SIZE = micro_batch_size
        _GRADIENT_ACCUMULATION_STEPS = gradient_accumulation_steps
        _SEQUENCE_LENGTH = sequence_length

    @staticmethod
    def get_local_batch_size(tracker_method: StepTrackerMethod) -> int:
        local_batch_size = _MICRO_BATCH_SIZE * _GRADIENT_ACCUMULATION_STEPS

        if tracker_method == StepTrackerMethod.tokens:
            local_batch_size = local_batch_size * _SEQUENCE_LENGTH

        return local_batch_size

    @staticmethod
    def get_global_batch_size(tracker_method: StepTrackerMethod) -> int:
        return StepTracker.get_local_batch_size(tracker_method) * ProcessGroupManager.get_data_parallel_world_size()
