from copy import deepcopy
from typing import Callable

from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
from deepspeed.ops.adam import DeepSpeedCPUAdam

from src.arguments import TrainingArgs
from src.constants import OptimizerKeys
from src.utils import register_profiler, register_timer
from src.utils.logging import warn_rank_0


@register_profiler("setup_optimizer")
@register_timer("setup_optimizer")
def get_optimizer(args: TrainingArgs, parameters: list) -> Callable:
    """setup optimizer for the model

    Args:
        args (TrainingArgs): training args
        parameters (list): list of model parameters

    Returns:
        Callable: an optimizer
    """

    optimizer_args = deepcopy(args.optimizer)
    optimizer_class = optimizer_args[OptimizerKeys.optimizer_class.value]
    del optimizer_args[OptimizerKeys.optimizer_class.value]

    if args.cpu_offload and optimizer_class not in [DeepSpeedCPUAdam, DeepSpeedCPUAdagrad]:
        warn_rank_0(
            "cpu offloading enabled with an unsupported optimizer, weird behaviour or performance drop might be observed"
        )

    optimizer = optimizer_class(parameters, **optimizer_args)
    return optimizer
