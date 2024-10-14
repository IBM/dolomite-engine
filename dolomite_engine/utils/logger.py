import logging
from warnings import warn

from ..containers import ModelContainer, OptimizerContainer
from .parallel import ProcessGroupManager, is_tracking_rank, run_rank_n


_LOGGER: logging.Logger = None


def set_logger(level: int = logging.INFO, colored_log: bool = False) -> None:
    stream = logging.StreamHandler()

    if colored_log:
        from .packages import is_colorlog_available

        assert is_colorlog_available(), "pip package colorlog is needed for colored logging"
        from colorlog import ColoredFormatter

        stream.setFormatter(ColoredFormatter("%(asctime)s - %(log_color)s[%(levelname)-8s] ▶%(reset)s %(message)s"))
        logging.basicConfig(level=level, handlers=[stream])
    else:
        logging.basicConfig(level=level, handlers=[stream], format="%(asctime)s - [%(levelname)-8s] ▶ %(message)s")

    global _LOGGER
    _LOGGER = logging.getLogger()


def get_logger() -> logging.Logger:
    return _LOGGER


@run_rank_n
def log_rank_0(level: int, msg: str) -> None:
    get_logger().log(level=level, msg=msg, stacklevel=3)


def log_metrics(level: int, msg: str) -> None:
    if not is_tracking_rank():
        return

    get_logger().log(level=level, msg=msg, stacklevel=3)


@run_rank_n
def print_rank_0(*args, **kwargs) -> None:
    """print on a single process"""

    print(*args, **kwargs)


def print_ranks_all(*args, **kwargs) -> None:
    """print on all processes sequentially, blocks other process and is slow. Please us sparingly."""

    for rank in range(ProcessGroupManager.get_world_size()):
        run_rank_n(print, rank=rank, barrier=True)(f"rank {rank}:", *args, **kwargs)


@run_rank_n
def warn_rank_0(*args, **kwargs) -> None:
    """warn on a single process"""

    warn(*args, **kwargs, stacklevel=3)


def log_model_optimizer_container(model_container: ModelContainer, optimizer_container: OptimizerContainer) -> None:
    """print model and optimizer

    Args:
        model_container (ModelContainer): container of models to print
        optimizer_container (OptimizerContainer): container of optimizers to print
    """

    log_rank_0(logging.INFO, "------------------------ model & optimizer list ------------------------")
    for model, optimizer in zip(model_container, optimizer_container):
        log_rank_0(logging.INFO, model)
        log_rank_0(logging.INFO, optimizer)
    log_rank_0(logging.INFO, "-------------------- end of model & optimizer list ---------------------")
