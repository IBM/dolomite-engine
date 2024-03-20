import logging
from warnings import warn

import wandb
from accelerate.tracking import WandBTracker
from aim import Run as AimRun
from pydantic import BaseModel
from tqdm import tqdm

from ..enums import ExperimentsTrackerName
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
    """experiments tracker for training"""

    def __init__(self, experiment: str, repo: str, experiments_tracker_name: ExperimentsTrackerName) -> None:
        self.experiments_tracker_name = experiments_tracker_name

        if experiments_tracker_name == ExperimentsTrackerName.aim:
            self.tracking_enabled = experiment is not None and repo is not None
            if self.tracking_enabled:
                self.run: AimRun = run_rank_n(AimRun)(experiment=experiment, repo=repo)
        elif experiments_tracker_name == ExperimentsTrackerName.wandb:
            self.tracking_enabled = repo is not None
            self.run = run_rank_n(wandb.init)(name=experiment, run_name=repo)
        else:
            raise ValueError(f"unexpected experiments_tracker ({experiments_tracker_name})")

        if not self.tracking_enabled:
            warn_rank_0("aim tracking is disabled since experiment_name or aim_repo was not specified")

    @run_rank_n
    def log_args(self, args: BaseModel) -> None:
        """log args

        Args:
            args (BaseModel): pydantic object
        """

        if self.tracking_enabled:
            args: dict = args.to_dict()

            if self.experiments_tracker_name == ExperimentsTrackerName.aim:
                for k, v in args.items():
                    try:
                        self.run[k] = v
                    except TypeError:
                        self.run[k] = str(v)
            elif self.experiments_tracker_name == ExperimentsTrackerName.wandb:
                wandb.config.update(args)
            else:
                raise ValueError(f"unexpected experiments_tracker ({self.experiments_tracker_name})")

    @run_rank_n
    def track(self, values: dict, step: int = None, context: str = None) -> None:
        """main tracking method

        Args:
            value (Any): value of the object to track
            step (int, optional): current step, auto-incremented if None. Defaults to None.
            context (str, optional): context for tracking. Defaults to None.
        """

        if self.tracking_enabled:
            if self.experiments_tracker_name == ExperimentsTrackerName.aim:
                if context is not None:
                    context = {"subset": context}

                for key, value in values.items():
                    self.run.track(value=value, name=key, step=step, context=context)
            elif self.experiments_tracker_name == ExperimentsTrackerName.wandb:
                if context is not None:
                    values = {f"{context}/{k}": v for k, v in values.items()}

                self.run.log(values=values, step=step)
            else:
                raise ValueError(f"unexpected experiments_tracker ({self.experiments_tracker_name})")

    @run_rank_n
    def finish(self) -> None:
        if self.tracking_enabled:
            if self.experiments_tracker_name == ExperimentsTrackerName.aim:
                self.run.close()
            elif self.experiments_tracker_name == ExperimentsTrackerName.wandb:
                self.run.finish()
            else:
                raise ValueError(f"unexpected experiments_tracker ({self.experiments_tracker_name})")


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
