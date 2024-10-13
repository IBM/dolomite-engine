import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ModelContainer:
    def __init__(self, model_list: list[nn.Module]) -> None:
        self.model_list = model_list

    def __iter__(self):
        for model in self.model_list:
            yield model

    def __getitem__(self, index: int) -> nn.Module:
        return self.model_list[index]

    def __setindex__(self, index: int, model: nn.Module) -> None:
        self.model_list[index] = model

    def train(self) -> "ModelContainer":
        for model in self:
            model.train()

    def eval(self) -> "ModelContainer":
        for model in self:
            model.eval()

        return self


class OptimizerContainer:
    def __init__(self, optimizer_list: list[Optimizer]) -> None:
        self.optimizer_list = optimizer_list

    def __iter__(self):
        for optimizer in self.optimizer_list:
            yield optimizer

    def zero_grad(self) -> None:
        for optimizer in self:
            optimizer.zero_grad()

    def step(self) -> None:
        for optimizer in self:
            optimizer.step()


class LRSchedulerContainer:
    def __init__(self, lr_scheduler_list: list[LambdaLR]) -> None:
        self.lr_scheduler_list = lr_scheduler_list

    def __iter__(self):
        for lr_scheduler in self.lr_scheduler_list:
            yield lr_scheduler

    def step(self) -> None:
        for lr_scheduler in self:
            lr_scheduler.step()
