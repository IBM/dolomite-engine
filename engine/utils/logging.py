import logging
import os
from typing import Any
from warnings import warn

from aim import Run
from aim.sdk.types import AimObject
from pydantic import BaseModel
from tqdm import tqdm

from .ranks import get_world_size, run_rank_n


class RunningMean:
    """tracks running mean for loss"""

    def __init__(self, window: int = 100) -> None:
        self.loss_list = []
        self.loss_sum = 0
        self.window = window

    @run_rank_n
    def add_loss(self, loss: float) -> float:
        """accumulates loss of the current step to the running mean

        Args:
            loss (float): loss at current step

        Returns:
            float: running mean of loss
        """

        self.loss_list.append(loss)
        self.loss_sum += loss

        if len(self.loss_list) > self.window:
            loss_remove = self.loss_list[0]
            self.loss_list = self.loss_list[1:]
            self.loss_sum -= loss_remove

        loss_mean = self.loss_sum / len(self.loss_list)
        return loss_mean


class ProgressBar:
    """progress bar for training or validation"""

    def __init__(self, start: int, end: int, desc: str = None) -> None:
        self.progress_bar: tqdm = run_rank_n(tqdm)(total=end, desc=desc)
        self.update(start)

    @run_rank_n
    def update(self, n: int = 1) -> None:
        """updates progress bar

        Args:
            n (int, optional): Number of steps to update the progress bar with. Defaults to 1.
        """

        self.progress_bar.update(n=n)

    @run_rank_n
    def track(self, **loss_kwargs) -> None:
        """track specific metrics in progress bar"""

        # for key in loss_kwargs:
        #     loss_kwargs[key] = "{0:.5f}".format(loss_kwargs[key])
        self.progress_bar.set_postfix(**loss_kwargs)


class ExperimentsTracker:
    """aim tracker for training"""

    def __init__(self, experiment: str, repo: str) -> None:
        self.tracking_enabled = self.is_tracking_enabled(experiment, repo)

        if self.tracking_enabled:
            run_rank_n(os.makedirs)(repo, exist_ok=True)
            self.run: Run = run_rank_n(Run)(experiment=experiment, repo=repo)
        else:
            warn_rank_0("aim tracking is disabled since experiment_name or aim_repo was not specified")

    @classmethod
    def is_tracking_enabled(cls, experiment: str, repo: str) -> bool:
        return experiment is not None and repo is not None

    @run_rank_n
    def log_args(self, args: BaseModel) -> None:
        """log args

        Args:
            args (BaseModel): pydantic object
        """

        if self.tracking_enabled:
            for k, v in vars(args).items():
                try:
                    self.run[k] = v
                except TypeError:
                    self.run[k] = str(v)

    @run_rank_n
    def track(
        self, value: Any, name: str = None, step: int = None, epoch: int = None, context: AimObject = None
    ) -> None:
        """main tracking method

        Args:
            value (Any): value of the object to track
            name (str, optional): name of the object to track. Defaults to None.
            step (int, optional): current step, auto-incremented if None. Defaults to None.
            epoch (int, optional): the training epoch. Defaults to None.
            context (AimObject, optional): context for tracking. Defaults to None.
        """

        if self.tracking_enabled:
            self.run.track(value=value, name=name, step=step, epoch=epoch, context=context)


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
    return get_logger().log(level=level, msg=msg, stacklevel=2)


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
