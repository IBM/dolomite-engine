import torch
from torch.cuda import Stream


_PARALLEL_COMPUTE_STREAM: Stream | None = None


class StreamManager:
    @staticmethod
    def get_parallel_compute_stream() -> Stream:
        global _PARALLEL_COMPUTE_STREAM

        if _PARALLEL_COMPUTE_STREAM is None:
            _PARALLEL_COMPUTE_STREAM = Stream()

        return _PARALLEL_COMPUTE_STREAM

    @staticmethod
    def get_default_compute_stream() -> Stream:
        return torch.cuda.default_stream()
