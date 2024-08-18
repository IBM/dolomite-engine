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

from ..enums import LRDecaySchedule, ParamsGroupMethod
from ..hf_models import GPTDolomiteConfig, GPTDolomiteForCausalLM, RNNDolomiteConfig, RNNDolomiteForCausalLM
from ..hf_models.modeling_utils import Attention
from ..hf_models.models.gpt_dolomite.layer import MLP
from ..model_wrapper import ModelWrapper
from ..utils import is_apex_available, is_deepspeed_available


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


def _get_param_groups(model: ModelWrapper, optimizer_class_args: dict, params_group_method: ParamsGroupMethod | None):
    if params_group_method is None:
        trainable_parameters_or_param_groups = model.parameters()
    elif params_group_method == ParamsGroupMethod.mup:
        assert isinstance(
            model.config, (GPTDolomiteConfig, RNNDolomiteConfig)
        ), "mup is not supported with this model architecture"
        assert isinstance(
            model.model, (GPTDolomiteForCausalLM, RNNDolomiteForCausalLM)
        ), "mup is not supported with this model architecture"
        assert (
            model.config.init_method == "mup"
        ), "both init method for model and params group method for optimizer should be set to mup"

        # collect parameters with mup learning rate
        mup_group = {}
        for module_name, module in model.named_modules():
            if isinstance(module, (Attention, MLP)):
                for param_name, param in module.named_parameters():
                    # we don't add bias to mup group
                    if not param_name.endswith("bias"):
                        # add name of module to name of subparam
                        mup_group[f"{module_name}.{param_name}"] = param

        # collect parameters without mup learning rate
        normal_group = []
        for param_name, param in model.named_parameters():
            if param_name not in mup_group:
                normal_group.append(param)

        assert len(normal_group) + len(mup_group) == len(
            list(model.parameters())
        ), "params in groups don't sum up to total parameters"

        trainable_parameters_or_param_groups = [
            {"params": normal_group},
            {"params": list(mup_group.values()), "lr": optimizer_class_args["lr"] / model.config.m_width},
        ]
    else:
        raise ValueError(f"unexpected params_group_method ({params_group_method})")

    return trainable_parameters_or_param_groups


def get_optimizer(
    optimizer_class_name: str,
    optimizer_class_args: dict,
    model: ModelWrapper,
    params_group_method: ParamsGroupMethod,
) -> Optimizer:
    """setup optimizer for the model

    Args:
        optimizer_class_name (str): optimizer class name
        optimizer_class_args (dict): args for the optimizer class
        model (ModelWrapper): model
        params_group_method (ParamsGroupMethod): the params grouping to use

    Returns:
        Optimizer: an optimizer
    """

    if optimizer_class_name not in _OPTIMIZER_CLASSES:
        raise ValueError(f"invalid class_name ({optimizer_class_name}) for optimizer")

    optimizer_class = _OPTIMIZER_CLASSES[optimizer_class_name]
    if optimizer_class is None:
        raise ImportError("relevant package for the optimizer is not installed")

    optimizer = optimizer_class(
        _get_param_groups(model, optimizer_class_args, params_group_method), **optimizer_class_args
    )
    return optimizer
