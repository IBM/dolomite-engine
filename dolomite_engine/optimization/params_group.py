import logging

import torch.nn as nn

from ..containers import ModelContainer
from ..enums import ParamsGroupMethod
from ..hf_models import (
    GPTDolomiteForCausalLM,
    GPTDolomiteForCausalLM_TP,
    LadderResidualForCausalLM,
    LadderResidualForCausalLM_TP,
)
from ..hf_models.modeling_utils import MLP, Attention, Mamba2, MoE
from ..model_wrapper import ModelWrapper
from ..utils import BaseArgs, log_rank_0


class _ParamsGroup(BaseArgs):
    name: str
    parameter_name_map: dict
    params_group_kwargs: dict = {}

    def to_param_group(self) -> dict:
        result = {}
        result.update(self.params_group_kwargs)

        # do in a sorted order
        param_names = self.get_param_names()

        result["params"] = []
        for param_name in param_names:
            result["params"].append(self.parameter_name_map[param_name])

        return result

    def get_param_names(self) -> list[str]:
        param_names = list(self.parameter_name_map.keys())
        param_names.sort()
        return param_names


class _ParamsGroupsList(BaseArgs):
    params_groups: list[_ParamsGroup] = []

    def add_params_group(self, params_group: _ParamsGroup) -> None:
        self.params_groups.append(params_group)

    def to_torch_compatible_params_groups(self) -> list[dict]:
        return [group.to_param_group() for group in self.params_groups]

    def get_param_names(self) -> list[str]:
        return {group.name: group.get_param_names() for group in self.params_groups}


def get_normal_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> list[_ParamsGroup]:
    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    normal_params = {}
    no_weight_decay_params = {}

    # remove layernorm and rmsnorm parameters from weight decay
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or module.__class__.__name__.lower().endswith("norm"):
            for param_name, param in module.named_parameters():
                no_weight_decay_params[f"{module_name}.{param_name}"] = param
        elif isinstance(module, Mamba2):
            for param_name, param in module.named_parameters():
                # we don't add bias or norms to mup group
                if param_name.endswith("A_log") or param_name.endswith("D"):
                    no_weight_decay_params[f"{module_name}.{param_name}"] = param

    # remove biases from weight decay
    for param_name, param in model.named_parameters():
        if param_name not in no_weight_decay_params and param_name.endswith("bias"):
            no_weight_decay_params[param_name] = param

    # these parameters have weight decay
    for param_name, param in model.named_parameters():
        if param_name not in no_weight_decay_params:
            normal_params[param_name] = param

    assert len(normal_params) + len(no_weight_decay_params) == len(
        list(model.parameters())
    ), "params in groups don't sum up to total parameters"

    if optimizer_class_args.get("weight_decay") == 0:
        no_weight_decay_params.update(normal_params)
        normal_params = {}

    params_group_list = _ParamsGroupsList()

    if len(normal_params) > 0:
        params_group_list.add_params_group(_ParamsGroup(name="normal", parameter_name_map=normal_params))
    if len(no_weight_decay_params) > 0:
        params_group_list.add_params_group(
            _ParamsGroup(
                name="no_weight_decay",
                parameter_name_map=no_weight_decay_params,
                params_group_kwargs={"weight_decay": 0},
            )
        )

    return params_group_list


def get_mup_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> list[_ParamsGroup]:
    assert isinstance(
        model.model,
        (
            GPTDolomiteForCausalLM,
            GPTDolomiteForCausalLM_TP,
            LadderResidualForCausalLM,
            LadderResidualForCausalLM_TP,
        ),
    ), "mup is not supported with this model architecture"

    assert (
        model.config.init_method == "mup"
    ), "both init method for model and params group method for optimizer should be set to mup"

    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    normal_params = {}
    no_weight_decay_params = {}
    mup_params = {}

    # collect parameters with mup learning rate
    for module_name, module in model.named_modules():
        if isinstance(module, (Attention, MLP, MoE)):
            for param_name, param in module.named_parameters():
                # we don't add bias or norms to mup group
                if not (param_name.endswith("bias") or "norm" in param_name):
                    # add name of module to name of subparam
                    mup_params[f"{module_name}.{param_name}"] = param
        elif isinstance(module, Mamba2):
            for param_name, param in module.named_parameters():
                if param_name in ["A_log", "D"] or not (param_name.endswith("bias") or "norm" in param_name):
                    mup_params[f"{module_name}.{param_name}"] = param
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or module.__class__.__name__.lower().endswith("norm"):
            for param_name, param in module.named_parameters():
                no_weight_decay_params[f"{module_name}.{param_name}"] = param

    # remove biases from weight decay
    for param_name, param in model.named_parameters():
        if param_name not in no_weight_decay_params and param_name.endswith("bias"):
            no_weight_decay_params[param_name] = param

    # collect parameters without mup learning rate
    for param_name, param in model.named_parameters():
        if param_name not in mup_params and param_name not in no_weight_decay_params:
            normal_params[param_name] = param

    assert len(normal_params) + len(no_weight_decay_params) + len(mup_params) == len(
        list(model.parameters())
    ), "params in groups don't sum up to total parameters"

    if optimizer_class_args.get("weight_decay") == 0:
        no_weight_decay_params.update(normal_params)
        normal_params = {}

    params_group_list = _ParamsGroupsList()

    if len(normal_params) > 0:
        params_group_list.add_params_group(_ParamsGroup(name="normal", parameter_name_map=normal_params))
    if len(no_weight_decay_params) > 0:
        params_group_list.add_params_group(
            _ParamsGroup(
                name="no_weight_decay",
                parameter_name_map=no_weight_decay_params,
                params_group_kwargs={"weight_decay": 0},
            )
        )
    if len(mup_params) > 0:
        params_group_list.add_params_group(
            _ParamsGroup(
                name="mup",
                parameter_name_map=mup_params,
                params_group_kwargs={"lr": optimizer_class_args["lr"] / model.config.m_width},
            )
        )

    return params_group_list


_PARAM_GROUPS = {
    None: get_normal_group_with_names,
    ParamsGroupMethod.mup: get_mup_group_with_names,
}


def get_param_groups_list(
    model_container: ModelContainer, optimizer_class_args: dict, params_group_method: ParamsGroupMethod | None
) -> list[list[_ParamsGroup]]:
    if params_group_method not in _PARAM_GROUPS:
        raise ValueError(f"unexpected `params_group_method` {params_group_method}")

    return [_PARAM_GROUPS[params_group_method](model, optimizer_class_args) for model in model_container]
