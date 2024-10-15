import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .model_wrapper import ModelWrapper
from .utils import log_rank_0


class _Container:
    model_list: list[ModelWrapper | Optimizer | LambdaLR]

    def __init__(self, model_list: list[ModelWrapper]) -> None:
        self.model_list = model_list

    def __iter__(self):
        for model in self.model_list:
            yield model

    def __getitem__(self, index: int) -> ModelWrapper:
        return self.model_list[index]

    def __setindex__(self, index: int, model: ModelWrapper) -> None:
        self.model_list[index] = model


class ModelContainer(_Container):
    def train(self) -> "ModelContainer":
        for model in self:
            model.train()

    def eval(self) -> "ModelContainer":
        for model in self:
            model.eval()

        return self


class LRSchedulerContainer(_Container):
    def step(self) -> None:
        for lr_scheduler in self:
            lr_scheduler.step()


class OptimizerContainer(LRSchedulerContainer):
    def zero_grad(self) -> None:
        for optimizer in self:
            optimizer.zero_grad()


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
