import logging
from typing import Any, Dict

import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from .utils import log_rank_0


class _Container(Stateful):
    def __init__(self, model_list: list[nn.Module]) -> None:
        self.model_list = model_list

    def __iter__(self):
        for model in self.model_list:
            yield model

    def __getitem__(self, index: int) -> nn.Module:
        return self.model_list[index]

    def __setindex__(self, index: int, model: nn.Module) -> None:
        self.model_list[index] = model


class ModelContainer(_Container):
    def train(self) -> "ModelContainer":
        for model in self:
            model.train()

    def eval(self) -> "ModelContainer":
        for model in self:
            model.eval()

        return self

    def state_dict(self) -> Dict[str, Any]:
        final_state_dict = {}

        for model in self:
            model_state_dict = get_model_state_dict(model)

            if model.has_teacher_model():
                model_state_dict = self._filter_out_teacher_state_dict(model_state_dict)

            final_state_dict.update(model_state_dict)

        return final_state_dict

    def _filter_out_teacher_state_dict(self, state_dict: dict) -> dict:
        result = {}
        for key, value in state_dict.items():
            if not "teacher_model" in key:
                result[key] = value

        return result


class LRSchedulerContainer(_Container):
    def step(self) -> None:
        for lr_scheduler in self:
            lr_scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        final_state_dict = []

        for lr_scheduler in self:
            lr_scheduler_state_dict = lr_scheduler.state_dict()
            final_state_dict.append(lr_scheduler_state_dict)

        return final_state_dict


class OptimizerContainer(LRSchedulerContainer):
    def zero_grad(self) -> None:
        for optimizer in self:
            optimizer.zero_grad()

    def state_dict(self, model_container: ModelContainer) -> Dict[str, Any]:
        final_state_dict = {}

        for model, optimizer in zip(model_container, self):
            optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
            final_state_dict.update(optimizer_state_dict)

        return final_state_dict


def log_model_optimizer_container(model_container: ModelContainer, optimizer_container: OptimizerContainer) -> None:
    """print model and optimizer

    Args:
        model_container (ModelContainer): container of models to print
        optimizer_container (OptimizerContainer): container of optimizers to print
    """

    log_rank_0(logging.INFO, "------------------------ model & optimizer list ------------------------")
    for model, optimizer in zip(model_container, optimizer_container):
        log_rank_0(logging.INFO, model)
        log_rank_0(logging.INFO, optimizer)
    log_rank_0(logging.INFO, "-------------------- end of model & optimizer list ---------------------")
