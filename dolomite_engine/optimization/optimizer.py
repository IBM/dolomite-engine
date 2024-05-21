import logging

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

from ..utils import is_apex_available, is_deepspeed_available, log_rank_0


if is_apex_available():
    from apex.optimizers import FusedAdam as ApexFusedAdam
    from apex.optimizers import FusedLAMB as ApexFusedLAMB
    from apex.optimizers import FusedNovoGrad as ApexFusedNovoGrad
    from apex.optimizers import FusedSGD as ApexFusedSGD
else:
    ApexFusedAdam = None
    ApexFusedLAMB = None
    ApexFusedNovoGrad = None
    ApexFusedSGD = None

if is_deepspeed_available():
    from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from deepspeed.ops.adam import FusedAdam as DeepSpeedFusedAdam
    from deepspeed.ops.lamb import FusedLamb as DeepSpeedFusedLAMB
    from deepspeed.runtime.fp16.onebit import OnebitAdam as DeepSpeedOnebitAdam
    from deepspeed.runtime.fp16.onebit import OnebitLamb as DeepSpeedOnebitLAMB
    from deepspeed.runtime.fp16.onebit import ZeroOneAdam as DeepSpeedZeroOneAdam
else:
    DeepSpeedCPUAdagrad = None
    DeepSpeedCPUAdam = None
    DeepSpeedFusedAdam = None
    DeepSpeedFusedLAMB = None
    DeepSpeedOnebitAdam = None
    DeepSpeedOnebitLAMB = None
    DeepSpeedZeroOneAdam = None


_OPTIMIZER_CLASSES = {
    # https://nvidia.github.io/apex/optimizers.html
    "ApexFusedAdam": ApexFusedAdam,
    "ApexFusedLAMB": ApexFusedLAMB,
    "ApexFusedNovoGrad": ApexFusedNovoGrad,
    "ApexFusedSGD": ApexFusedSGD,
    # https://deepspeed.readthedocs.io/en/latest/optimizers.html
    "DeepSpeedCPUAdagrad": DeepSpeedCPUAdagrad,
    "DeepSpeedCPUAdam": DeepSpeedCPUAdam,
    "DeepSpeedFusedAdam": DeepSpeedFusedAdam,
    "DeepSpeedFusedLAMB": DeepSpeedFusedLAMB,
    "DeepSpeedOnebitAdam": DeepSpeedOnebitAdam,
    "DeepSpeedOnebitLAMB": DeepSpeedOnebitLAMB,
    "DeepSpeedZeroOneAdam": DeepSpeedZeroOneAdam,
    # https://pytorch.org/docs/stable/optim.html
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


def get_optimizer(
    optimizer_class_name: str, optimizer_class_args: dict, cpu_offload: bool, parameters: list
) -> Optimizer:
    """setup optimizer for the model

    Args:
        optimizer_class_name (str): optimizer class name
        optimizer_class_args (dict): args for the optimizer class
        parameters (list): list of model parameters

    Returns:
        Optimizer: an optimizer
    """

    if optimizer_class_name not in _OPTIMIZER_CLASSES:
        raise ValueError(f"invalid class_name ({optimizer_class_name}) for optimizer")

    optimizer_class = _OPTIMIZER_CLASSES[optimizer_class_name]
    if optimizer_class is None:
        raise ImportError("relevant package for the optimizer is not installed")

    if cpu_offload and optimizer_class not in [DeepSpeedCPUAdam, DeepSpeedCPUAdagrad]:
        log_rank_0(
            logging.WARN,
            "cpu offloading enabled with an unsupported optimizer, weird behaviour or performance drop might be observed",
        )

    optimizer = optimizer_class(parameters, **optimizer_class_args)
    return optimizer
