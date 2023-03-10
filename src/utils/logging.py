import logging
import os
from argparse import Namespace
from typing import Any
from warnings import warn

from aim import Run
from tqdm import tqdm

from src.utils.distributed import get_world_size, run_rank_n


@run_rank_n
def print_rank_0(*args, **kwargs) -> None:
    print(*args, **kwargs)


def print_ranks_all(*args, **kwargs) -> None:
    for rank in range(get_world_size()):
        run_rank_n(print, rank=rank, barrier=True)(f"rank {rank}:", *args, **kwargs)


@run_rank_n
def warn_rank_0(*args, **kwargs) -> None:
    warn(*args, **kwargs)


class RunningMean:
    def __init__(self, window: int = 100) -> None:
        self.loss_list = []
        self.loss_sum = 0
        self.window = window

    @run_rank_n
    def add_loss(self, loss: float) -> None:
        self.loss_list.append(loss)
        self.loss_sum += loss

        if len(self.loss_list) > self.window:
            loss_remove = self.loss_list[0]
            self.loss_list = self.loss_list[1:]
            self.loss_sum -= loss_remove

        loss_mean = self.loss_sum / len(self.loss_list)
        return loss_mean


class ProgressBar:
    def __init__(self, start: int, end: int, desc: str = None) -> None:
        self.progress_bar: tqdm = run_rank_n(tqdm)(total=end, desc=desc)
        self.update(start)

    @run_rank_n
    def update(self, n: int = 1) -> None:
        self.progress_bar.update(n=n)

    @run_rank_n
    def track(self, **loss_kwargs) -> None:
        # for key in loss_kwargs:
        #     loss_kwargs[key] = "{0:.5f}".format(loss_kwargs[key])
        self.progress_bar.set_postfix(**loss_kwargs)


class AimTracker:
    def __init__(self, experiment: str, repo: str, disable: bool = False) -> None:
        if disable:
            self.run = None
        else:
            run_rank_n(os.makedirs)(repo, exist_ok=True)
            self.run = run_rank_n(Run)(experiment=experiment, repo=repo)

    @run_rank_n
    def log_hyperparam(self, name: str, value: Any) -> None:
        if self.run is None:
            return

        try:
            self.run[name] = value
        except TypeError:
            self.run[name] = str(value)

    @run_rank_n
    def log_args(self, args: Namespace) -> None:
        if self.run is None:
            return

        for k, v in vars(args).items():
            self.log_hyperparam(k, v)

    @run_rank_n
    def track(self, *args, **kwargs) -> None:
        if self.run is None:
            return

        self.run.track(*args, **kwargs)


class Logger:
    def __init__(
        self,
        name: str,
        logfile: str,
        level: int = logging.INFO,
        format: str = "%(asctime)s - [%(levelname)s]: %(message)s",
        # format: str = "%(asctime)s - [%(levelname)s] - (%(filename)s:%(lineno)d): %(message)s",
        disable_stdout: bool = False,
    ) -> None:
        dirname = os.path.dirname(logfile)
        if dirname is not None and dirname not in ["", "."]:
            run_rank_n(os.makedirs)(dirname, exist_ok=True)

        # for writing to file
        handlers = [run_rank_n(logging.FileHandler)(logfile, "a")]
        # for writing to stdout
        if not disable_stdout:
            handlers.append(run_rank_n(logging.StreamHandler)())

        run_rank_n(logging.basicConfig)(level=level, handlers=handlers, format=format)

        self.logger: logging.Logger = run_rank_n(logging.getLogger)(name)
        self.stacklevel = 3

    @run_rank_n
    def info(self, msg) -> None:
        self.logger.info(msg, stacklevel=self.stacklevel)

    @run_rank_n
    def debug(self, msg) -> None:
        self.logger.debug(msg, stacklevel=self.stacklevel)

    @run_rank_n
    def critical(self, msg) -> None:
        self.logger.critical(msg, stacklevel=self.stacklevel)

    @run_rank_n
    def error(self, msg) -> None:
        self.logger.error(msg, stacklevel=self.stacklevel)

    @run_rank_n
    def warn(self, msg) -> None:
        self.logger.warn(msg, stacklevel=self.stacklevel)


class ExperimentsTracker(Logger, AimTracker):
    def __init__(
        self,
        logger_name: str,
        experiment_name: str,
        aim_repo: str,
        disable_aim: bool,
    ) -> None:
        Logger.__init__(self, logger_name, os.path.join(aim_repo, f"{experiment_name}.log"), disable_stdout=True)
        AimTracker.__init__(self, experiment_name, aim_repo, disable_aim)
