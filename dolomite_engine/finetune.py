import contextlib
import logging

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training, save_checkpoint
from .data import DataLoader, get_dataloader, infinite_iterator
from .distributed import wrap_model_for_distributed_training
from .enums import DatasetSplit, DistributedBackend, Mode
from .model_wrapper import ModelWrapperForFinetuning, get_model, log_model
from .utils import (
    ExperimentsTracker,
    RunningMean,
    init_distributed,
    log_rank_0,
    register_profiler,
    register_timer,
    setup_tf32,
)


def track_train_metrics(
    global_step: int,
    train_loss_step: float,
    current_lr: float,
    experiments_tracker: ExperimentsTracker,
    loss_running_mean_tracker: RunningMean,
    flops: int = None,
    tokens_per_day: int = None,
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        train_loss_step (float): training loss at the current step
        current_lr (float): learning rate at the current step
        experiments_tracker (ExperimentsTracker): metrics tracker
        loss_running_mean_tracker (RunningMean): running mean accumulator for loss
        flops (int, optional): total model flops. Defaults to None
    """

    # update loss running mean
    loss_running_mean = loss_running_mean_tracker.add_loss(train_loss_step)

    message = {"loss_step": train_loss_step, "loss_running_mean": loss_running_mean, "learning_rate": current_lr}
    if flops is not None:
        message["FLOPS"] = flops
    experiments_tracker.track(message, step=global_step, context="train")

    message = (
        f"step = {global_step}, train_loss (batch) = {train_loss_step}, "
        f"train_loss (running_mean) = {loss_running_mean}, "
        f"learning_rate = {current_lr}, "
    )
    if flops is not None:
        message += f"FLOPS = {flops}, "

    if tokens_per_day is not None:
        message += f"throughput = {tokens_per_day}B tokens per day"

    log_rank_0(logging.INFO, message)


def track_val_metrics(global_step: int, val_loss: float, experiments_tracker: ExperimentsTracker) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        val_loss (float): validation loss for the validation data
        experiments_tracker (ExperimentsTracker): metrics tracker
    """

    log_rank_0(logging.INFO, f"step = {global_step}, val_loss = {val_loss}")
    experiments_tracker.track({"loss": val_loss}, step=global_step, context="val")


@register_profiler("train_step")
@register_timer("train_step")
def train_step(
    model: ModelWrapperForFinetuning,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    distributed_backend: DistributedBackend,
    train_dataloader: DataLoader,
    gradient_accumulation_steps: int,
    gradient_clipping: float,
) -> float:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model (ModelWrapperForFinetuning): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LamdaLR): learning rate scheduler
        distributed_backend (DistributedBackend): distributed backend
        train_dataloader (DataLoader): training dataloader
        gradient_accumulation_steps (int): gradient accumulation steps
        gradient_clipping (float): gradient clipping value

    Returns:
        float: loss at the current step
    """

    no_sync = model.no_sync if distributed_backend == DistributedBackend.torch else contextlib.nullcontext
    loss = 0
    if distributed_backend == DistributedBackend.torch:
        optimizer.zero_grad()

    with no_sync():
        for _ in range(gradient_accumulation_steps - 1):
            batch = next(train_dataloader)
            loss_micro_step = model(batch)
            loss += loss_micro_step

            # compute gradients
            if distributed_backend == DistributedBackend.deepspeed:
                model.backward(loss_micro_step)
                model.step()
            elif distributed_backend == DistributedBackend.torch:
                loss_micro_step.backward()
            else:
                raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    batch = next(train_dataloader)
    loss_micro_step = model(batch)
    loss += loss_micro_step

    # compute gradients
    if distributed_backend == DistributedBackend.deepspeed:
        model.backward(loss_micro_step)
        model.step()
    elif distributed_backend == DistributedBackend.torch:
        loss_micro_step.backward()

        if gradient_clipping is not None:
            model.clip_grad_norm_(gradient_clipping)

        optimizer.step()
        lr_scheduler.step()
    else:
        raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    loss = loss / gradient_accumulation_steps
    loss = loss.item()

    return loss


def train(
    args: TrainingArgs,
    model: ModelWrapperForFinetuning,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    experiments_tracker: ExperimentsTracker,
    starting_iteration: int = 0,
) -> None:
    """main training loop for the program

    Args:
        args (TrainingArgs): training args
        model (ModelWrapperForFinetuning): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LRScheduler): learning rate scheduler
        train_dataloader (DataLoader): training dataloader
        val_dataloader (DataLoader): validation dataloader
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

    loss_running_mean_tracker = RunningMean()

    model.train()

    # need this for iterating infinitely
    train_dataloader_infinite = infinite_iterator(train_dataloader)

    if eval_during_training:
        evaluate(val_dataloader, model, starting_iteration, experiments_tracker)

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1

        loss_step = train_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            distributed_backend=distributed_backend,
            train_dataloader=train_dataloader_infinite,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
        )

        if global_step % log_interval == 0:
            track_train_metrics(
                global_step=global_step,
                train_loss_step=loss_step,
                current_lr=model.lr_scheduler.get_lr()[0]
                if distributed_backend == DistributedBackend.deepspeed
                else lr_scheduler.get_lr()[0],
                experiments_tracker=experiments_tracker,
                loss_running_mean_tracker=loss_running_mean_tracker,
            )

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(val_dataloader, model, global_step, experiments_tracker)

        if global_step % save_interval == 0 or global_step == num_training_steps:
            save_checkpoint(args, model, optimizer, lr_scheduler, train_dataloader, experiments_tracker, global_step)


@register_profiler("evaluate_dataset")
@torch.no_grad()
def evaluate(
    val_dataloader: DataLoader,
    model: ModelWrapperForFinetuning,
    global_step: int,
    experiments_tracker: ExperimentsTracker,
) -> float:
    """main validation loop for the program

    Args:
        val_dataloader (DataLoader): validation dataloader
        model (ModelWrapperForFinetuning): model
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker

    Returns:
        float: loss at the current step
    """

    if val_dataloader is None:
        return

    model.eval()

    loss_sum = 0
    micro_step = 0

    for batch in val_dataloader:
        loss_value = model(batch).item()
        loss_sum += loss_value
        micro_step += 1

    loss_mean = loss_sum / micro_step
    track_val_metrics(global_step, loss_mean, experiments_tracker)

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

    train_dataloader = get_dataloader(
        args,
        split=DatasetSplit.train,
        mode=mode,
        tokenizer=model.tokenizer,
        is_encoder_decoder=model.is_encoder_decoder,
        padding_side=model.padding_side,
    )

    val_dataloader = None
    if args.training_parameters.eval_during_training:
        val_dataloader = get_dataloader(
            args,
            split=DatasetSplit.val,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
            padding_side=model.padding_side,
        )

    model, optimizer, lr_scheduler = wrap_model_for_distributed_training(args, model)

    log_model(model)

    starting_iteration = 0
    experiments_tracker_state_dict = None
    if args.load_args is not None:
        starting_iteration, _, experiments_tracker_state_dict = load_checkpoint_for_training(
            args, model, optimizer, lr_scheduler, train_dataloader
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
        val_dataloader=val_dataloader,
        experiments_tracker=experiments_tracker,
        starting_iteration=starting_iteration,
    )


if __name__ == "__main__":
    main()
