import logging
from warnings import warn

from .ranks import get_world_size, run_rank_n


_LOGGER: logging.Logger = None


def set_logger(level: int = logging.INFO) -> None:
    # format: str = "%(asctime)s - [%(levelname)s]: %(message)s"
    logging.basicConfig(
        level=level,
        handlers=[logging.StreamHandler()],
        format="%(asctime)s - [%(levelname)s] - (%(filename)s:%(lineno)d): %(message)s",
    )

    global _LOGGER
    _LOGGER = run_rank_n(logging.getLogger)()


def get_logger() -> logging.Logger:
    assert _LOGGER is not None
    return _LOGGER


@run_rank_n
def log_rank_0(level: int, msg: str) -> None:
    return get_logger().log(level=level, msg=msg, stacklevel=3)


@run_rank_n
def print_rank_0(*args, **kwargs) -> None:
    """print on a single process"""

    print(*args, **kwargs)


def print_ranks_all(*args, **kwargs) -> None:
    """print on all processes sequentially, blocks other process and is slow. Please us sparingly."""

    for rank in range(get_world_size()):
        run_rank_n(print, rank=rank, barrier=True)(f"rank {rank}:", *args, **kwargs)


@run_rank_n
def warn_rank_0(*args, **kwargs) -> None:
    """warn on a single process"""

    warn(*args, **kwargs)
