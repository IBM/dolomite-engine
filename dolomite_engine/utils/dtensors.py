from contextlib import contextmanager


_USE_DTENSORS_FOR_COMPUTATIONS: bool = False


@contextmanager
def enable_dtensors_for_computation(enable: bool = True):
    global _USE_DTENSORS_FOR_COMPUTATIONS

    original_value = _USE_DTENSORS_FOR_COMPUTATIONS
    _USE_DTENSORS_FOR_COMPUTATIONS = enable

    yield

    _USE_DTENSORS_FOR_COMPUTATIONS = original_value


def is_dtensors_computation_enabled() -> bool:
    global _USE_DTENSORS_FOR_COMPUTATIONS
    return _USE_DTENSORS_FOR_COMPUTATIONS
