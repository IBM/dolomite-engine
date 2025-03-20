from ..containers import ModelContainer, OptimizerContainer
from .optimizer import get_optimizer_container
from .scheduler import get_scheduler_container


def get_learning_rate(
    model_container: ModelContainer, lr_scheduler_container: OptimizerContainer, use_optimizer_with_backward_hook: bool
) -> float:
    if use_optimizer_with_backward_hook:
        parameters = model_container[0].parameters()
        param = next(parameters)

        lr = param._lr_scheduler.get_lr()[0]
    else:
        lr = lr_scheduler_container[0].get_lr()[0]

    return lr
