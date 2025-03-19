from functools import partial

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.adadelta import Adadelta as TorchAdadelta
from torch.optim.adagrad import Adagrad as TorchAdagrad
from torch.optim.adam import Adam as TorchAdam
from torch.optim.adamax import Adamax as TorchAdamax
from torch.optim.adamw import AdamW as TorchAdamW
from torch.optim.asgd import ASGD as TorchASGD
from torch.optim.lbfgs import LBFGS as TorchLBFGS
from torch.optim.nadam import NAdam as TorchNAdam
from torch.optim.radam import RAdam as TorchRAdam
from torch.optim.rmsprop import RMSprop as TorchRMSprop
from torch.optim.rprop import Rprop as TorchRprop
from torch.optim.sgd import SGD as TorchSGD

from ..containers import BackwardHookOptimizerContainer, ModelContainer, OptimizerContainer
from ..enums import ParamsGroupMethod
from .params_group import get_param_groups_list


# https://pytorch.org/docs/stable/optim.html
_OPTIMIZER_CLASSES = {
    "TorchAdadelta": TorchAdadelta,
    "TorchAdagrad": TorchAdagrad,
    "TorchAdam": TorchAdam,
    "TorchAdamax": TorchAdamax,
    "TorchAdamW": TorchAdamW,
    "TorchASGD": TorchASGD,
    "TorchLBFGS": TorchLBFGS,
    "TorchNAdam": TorchNAdam,
    "TorchRAdam": TorchRAdam,
    "TorchRMSprop": TorchRMSprop,
    "TorchRprop": TorchRprop,
    "TorchSGD": TorchSGD,
}


def get_optimizer_container(
    optimizer_class_name: str,
    optimizer_class_args: dict,
    model_container: ModelContainer,
    params_group_method: ParamsGroupMethod,
    use_optimizer_with_backward_hook: bool,
) -> OptimizerContainer:
    """setup list of optimizers for the model

    Args:
        optimizer_class_name (str): optimizer class name
        optimizer_class_args (dict): args for the optimizer class
        model_container (ModelContainer): model container
        params_group_method (ParamsGroupMethod): the params grouping to use
        use_optimizer_with_backward_hook (bool): whether to use optimizer as a backward hook

    Returns:
        OptimizerContainer: optimizer container
    """

    if optimizer_class_name not in _OPTIMIZER_CLASSES:
        raise ValueError(f"invalid class_name ({optimizer_class_name}) for optimizer")

    optimizer_class = _OPTIMIZER_CLASSES[optimizer_class_name]
    if optimizer_class is None:
        raise ImportError("relevant package for the optimizer is not installed")

    params_groups_list = get_param_groups_list(model_container, optimizer_class_args, params_group_method)
    optimizer_list = []

    if use_optimizer_with_backward_hook:
        for model, params_groups in zip(model_container, params_groups_list):
            optimizer_map = {}

            for group in params_groups:
                params = group.pop("params")
                for param in params:
                    optimizer_map[param] = optimizer_class([param], **group)

            for param in model.parameters():

                def _step(p: nn.Parameter, optimizer: Optimizer) -> None:
                    optimizer.step()
                    optimizer.zero_grad()

                param.register_post_accumulate_grad_hook(partial(_step, optimizer=optimizer_map[param]))

            optimizer_list.append(optimizer_map)

        optimizer_list = BackwardHookOptimizerContainer(optimizer_list)
    else:
        for params_groups in params_groups_list:
            torch_compatible_params_group = []
            for params_group in params_groups:
                torch_compatible_params_group.append(params_group.to_param_group())

            optimizer_list.append(optimizer_class(torch_compatible_params_group))

        optimizer_list = OptimizerContainer(optimizer_list)

    return optimizer_list
