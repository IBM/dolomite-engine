from typing import Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..enums import LRDecaySchedule, ParamsGroupMethod
from ..hf_models import GPTDolomiteConfig, GPTDolomiteForCausalLM
from ..hf_models.modeling_utils import Attention
from ..hf_models.models.gpt_dolomite.layer import MLP
from ..model_wrapper import ModelWrapper
from .optimizer import get_optimizer
from .scheduler import get_scheduler


def get_optimizer_and_lr_scheduler(
    optimizer_class_name: str,
    optimizer_class_args: dict,
    cpu_offload: bool,
    model: ModelWrapper,
    num_warmup_steps: int,
    num_constant_steps: int,
    num_decay_steps: int,
    num_training_steps: int,
    lr_decay_style: LRDecaySchedule,
    lr_decay_factor: float,
    params_group_method: ParamsGroupMethod,
    extra_lr_scheduler_args: dict,
) -> Tuple[Optimizer, LambdaLR]:
    trainable_parameters_or_param_groups = _get_param_groups(model, optimizer_class_args, params_group_method)

    optimizer = get_optimizer(
        optimizer_class_name=optimizer_class_name,
        optimizer_class_args=optimizer_class_args,
        cpu_offload=cpu_offload,
        parameters=trainable_parameters_or_param_groups,
    )

    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_constant_steps=num_constant_steps,
        num_decay_steps=num_decay_steps,
        num_training_steps=num_training_steps,
        lr_decay_style=lr_decay_style,
        lr_decay_factor=lr_decay_factor,
        extra_lr_scheduler_args=extra_lr_scheduler_args,
    )

    return optimizer, lr_scheduler


def _get_param_groups(model: ModelWrapper, optimizer_class_args: dict, params_group_method: ParamsGroupMethod):
    if params_group_method is None:
        trainable_parameters_or_param_groups = model.parameters()
    elif params_group_method == ParamsGroupMethod.mup:
        assert isinstance(model.config, GPTDolomiteConfig), "mup is only supported with GPTDolomiteForCausalLM"
        assert isinstance(model.model, GPTDolomiteForCausalLM), "mup is only supported with GPTDolomiteForCausalLM"
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
