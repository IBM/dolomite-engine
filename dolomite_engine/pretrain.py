import logging
import time
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training, save_checkpoint
from .data import get_megatron_gpt_dataloaders
from .distributed import wrap_model_for_distributed_training
from .enums import DistributedBackend, Mode
from .finetune import track_train_metrics, train_step
from .model_wrapper import ModelWrapperForPretraining, get_model, log_model
from .utils import (
    ExperimentsTracker,
    RunningMean,
    get_world_size,
    init_distributed,
    log_rank_0,
    register_profiler,
    setup_tf32,
)


def track_val_metrics(
    global_step: int, val_loss: float, experiments_tracker: ExperimentsTracker, group_name: str = None
) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        val_loss (float): validation loss for the validation data
        experiments_tracker (ExperimentsTracker): metrics tracker
        group_name (str): group name for the validation / test set
    """

    message = f"step = {global_step}, val_loss = {val_loss:.4f}"
    if group_name is not None:
        message += f", group_name = {group_name}"

    log_rank_0(logging.INFO, message)
    experiments_tracker.track(
        {"loss" if group_name is None else f"loss-{group_name}": val_loss}, step=global_step, context="val"
    )


def train(
    args: TrainingArgs,
    model: ModelWrapperForPretraining,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: DataLoader,
    val_dataloaders: List[DataLoader],
    test_dataloaders: List[DataLoader],
    experiments_tracker: ExperimentsTracker,
    starting_iteration: int = 0,
) -> None:
    """main training loop for the program

    Args:
        args (TrainingArgs): training args
        model (ModelWrapperForPretraining): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LRScheduler): learning rate scheduler
        train_dataloader (DataLoader): training dataloader
        val_dataloaders (List[DataLoader]): validation dataloaders
        test_dataloaders (List[DataLoader]): test dataloaders
        experiments_tracker (ExperimentsTracker): metrics tracker
        starting_iteration (int): starting iteration
    """

    num_training_steps = args.training_parameters.num_training_steps
    gradient_accumulation_steps = args.training_parameters.gradient_accumulation_steps
    gradient_clipping = args.training_parameters.gradient_clipping

    eval_during_training = args.training_parameters.eval_during_training
    eval_interval = args.training_parameters.eval_interval
    distributed_backend = args.distributed_args.distributed_backend
    save_interval = args.save_args.save_interval
    log_interval = args.logging_args.log_interval

    val_weighted_split_paths = args.datasets[0].class_args.get("val_weighted_split_paths")
    group_names = [None]
    if val_weighted_split_paths is not None:
        group_names = [key for key in val_weighted_split_paths.keys()[0]]

    loss_running_mean_tracker = RunningMean()

    model.train()

    if eval_during_training:
        eval_steps = args.datasets[0].class_args.get("eval_steps")
        evaluate(val_dataloaders, model, starting_iteration, experiments_tracker, eval_steps, group_names)

    micro_batch_size = args.training_parameters.micro_batch_size
    sequence_length = args.datasets[0].class_args.get("sequence_length")
    model_flops = model.get_model_tflops(micro_batch_size * gradient_accumulation_steps, sequence_length)
    tokens_per_batch = micro_batch_size * gradient_accumulation_steps * get_world_size() * sequence_length

    start_time = time.perf_counter()
    steps_since_start_time = 0

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1
        steps_since_start_time += 1

        loss_step, grad_norm_step = train_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            distributed_backend=distributed_backend,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
        )

        if global_step % log_interval == 0:
            time_elapsed = time.perf_counter() - start_time
            step_time = time_elapsed / steps_since_start_time

            track_train_metrics(
                global_step=global_step,
                train_loss_step=loss_step,
                grad_norm_step=grad_norm_step,
                current_lr=model.lr_scheduler.get_lr()[0]
                if distributed_backend == DistributedBackend.deepspeed
                else lr_scheduler.get_lr()[0],
                experiments_tracker=experiments_tracker,
                loss_running_mean_tracker=loss_running_mean_tracker,
                flops=None if model_flops is None else model_flops * steps_since_start_time / time_elapsed,
                billion_tokens_per_day=tokens_per_batch * 86400 / step_time / 1e9,
                step_time=step_time,
            )
            start_time = time.perf_counter()
            steps_since_start_time = 0

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(val_dataloaders, model, global_step, experiments_tracker, eval_steps, group_names)

        if global_step % save_interval == 0 or global_step == num_training_steps:
            save_checkpoint(
                args,
                model,
                optimizer,
                lr_scheduler,
                None,
                experiments_tracker,
                global_step,
                {"consumed_samples": global_step * micro_batch_size * gradient_accumulation_steps * get_world_size()},
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0

    if eval_during_training:
        evaluate(test_dataloaders, model, global_step, experiments_tracker, eval_steps, group_names)


@register_profiler("evaluate_dataset")
@torch.no_grad()
def evaluate(
    val_dataloaders: List[DataLoader],
    model: ModelWrapperForPretraining,
    global_step: int,
    experiments_tracker: ExperimentsTracker,
    eval_steps: int,
    group_names: List[str],
) -> float:
    """main validation loop for the program

    Args:
        val_dataloaders (List[DataLoader]): list of validation dataloaders
        model (ModelWrapperForPretraining): model
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        eval_steps (int): number of steps to run eval for
        group_names (List[str]): names of the datasets in validation/test group

    Returns:
        float: loss at the current step
    """

    if val_dataloaders is None or len(val_dataloaders) == 0:
        return

    model.eval()

    for group_name, val_dataloader in zip(group_names, val_dataloaders):
        loss_sum = 0
        for _ in range(eval_steps):
            batch = next(val_dataloader)
            loss_value = model(batch).item()
            loss_sum += loss_value

        loss_mean = loss_sum / eval_steps
        track_val_metrics(global_step, loss_mean, experiments_tracker, group_name)

    model.train()

    return loss_mean


def main() -> None:
    """main program"""

    mode = Mode.training

    setup_tf32()

    args: TrainingArgs = get_args(mode)

    # initialize distributed with nccl for multi-node communications
    init_distributed()
    set_seed(args.random_args.seed)

    model = get_model(args, mode)
    model, optimizer, lr_scheduler = wrap_model_for_distributed_training(args, model)

    log_model(model)

    starting_iteration = 0
    metadata = None
    experiments_tracker_state_dict = None
    if args.load_args is not None:
        starting_iteration, metadata, experiments_tracker_state_dict = load_checkpoint_for_training(
            args, model, optimizer, lr_scheduler, None
        )

    train_dataloader, val_dataloaders, test_dataloaders = get_megatron_gpt_dataloaders(
        args, model.tokenizer, 0 if metadata is None else metadata["consumed_samples"]
    )

    experiments_tracker = ExperimentsTracker(
        args.logging_args.experiments_tracker_name,
        args.logging_args.aim_args,
        args.logging_args.wandb_args,
        checkpoint_metadata=experiments_tracker_state_dict,
    )
    # track all hyperparams in args
    experiments_tracker.log_args(args)

    # main training loop
    train(
        args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        test_dataloaders=test_dataloaders,
        experiments_tracker=experiments_tracker,
        starting_iteration=starting_iteration,
    )


if __name__ == "__main__":
    main()
