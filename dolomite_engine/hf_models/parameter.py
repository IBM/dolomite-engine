import torch.nn as nn


def mark_parameter_as_no_weight_decay(parameter: nn.Parameter | None) -> nn.Parameter | None:
    if parameter is not None:
        parameter._no_weight_decay = True

    return parameter


def mark_parameter_as_mup_learning_rate(parameter: nn.Parameter | None) -> nn.Parameter | None:
    if parameter is not None:
        parameter._has_mup_learning_rate = True

    return parameter
