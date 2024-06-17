import contextlib
import logging
from typing import Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training, save_checkpoint
from .communication import Communication
from .data import ResumableDataLoader, get_dataloader, get_next_batch, infinite_iterator
from .distributed import wrap_model_for_distributed_training
from .enums import DatasetSplit, DistributedBackend, FP8Backend, Mode
from .model_wrapper import ModelWrapperForFinetuning, get_model, log_model
from .utils import (
    ExperimentsTracker,
    ProcessGroupManager,
    RunningMean,
    init_distributed,
    is_transformer_engine_available,
    log_rank_0,
    setup_tf32,
)


if is_transformer_engine_available():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format


def track_train_metrics(
    global_step: int,
    train_loss_step: float,
    grad_norm_step: float,
    current_lr: float,
    experiments_tracker: ExperimentsTracker,
    loss_running_mean_tracker: RunningMean,
    flops: float = None,
    billion_tokens_per_day: float = None,
    step_time: float = None,
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        train_loss_step (float): training loss at the current step
        current_lr (float): learning rate at the current step
        experiments_tracker (ExperimentsTracker): metrics tracker
        loss_running_mean_tracker (RunningMean): running mean accumulator for loss
        flops (float, optional): total model flops. Defaults to None
        billion_tokens_per_day (float, optional): billions of tokens per day. Defaults to None
        step_time (float, optional): time per step in seconds
    """

    # update loss running mean
    loss_running_mean = loss_running_mean_tracker.add_loss(train_loss_step)

    # experiments tracker
    message = {"loss_step": train_loss_step, "loss_running_mean": loss_running_mean, "learning_rate": current_lr}

    if grad_norm_step is not None:
        message["grad_norm"] = grad_norm_step

    if flops is not None:
        message["FLOPS"] = flops

    if billion_tokens_per_day is not None:
        message["throughput (B tokens/day)"] = billion_tokens_per_day

    if step_time is not None:
        message["step time (sec)"] = step_time

    experiments_tracker.track(message, step=global_step, context="train")

    # terminal
    message = (
        f"step = {global_step}, train_loss (batch) = {train_loss_step:.4f}, "
        f"train_loss (running_mean) = {loss_running_mean:.4f}, "
        f"learning_rate = {current_lr:.3E}"
    )

    if grad_norm_step is not None:
        message += f", grad_norm = {grad_norm_step:.2f}"

    if flops is not None:
        message += f", FLOPS = {flops:.2f}"

    if billion_tokens_per_day is not None:
        message += f", throughput = {billion_tokens_per_day:.2f} B tokens/day"

    if step_time is not None:
        message += f", step_time = {step_time:.3f} sec"

    log_rank_0(logging.INFO, message)


def track_val_metrics(global_step: int, val_loss: float, experiments_tracker: ExperimentsTracker) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        val_loss (float): validation loss for the validation data
        experiments_tracker (ExperimentsTracker): metrics tracker
    """

    log_rank_0(logging.INFO, f"step = {global_step}, val_loss = {val_loss:.4f}")
    experiments_tracker.track({"loss": val_loss}, step=global_step, context="val")


def train_step(
    model: ModelWrapperForFinetuning,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    distributed_backend: DistributedBackend,
    train_dataloader: ResumableDataLoader,
    gradient_accumulation_steps: int,
    gradient_clipping: float,
    train_step_context: contextlib.AbstractContextManager,
) -> Tuple[float, float]:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model (ModelWrapperForFinetuning): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LamdaLR): learning rate scheduler
        distributed_backend (DistributedBackend): distributed backend
        train_dataloader (ResumableDataLoader): training dataloader
        gradient_accumulation_steps (int): gradient accumulation steps
        gradient_clipping (float): gradient clipping value

    Returns:
        Tuple[float, float]: loss at the current step, grad norm at the current step
    """

    no_sync = contextlib.nullcontext
    if distributed_backend == DistributedBackend.torch:
        # FSDP-2
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(False)
        else:
            no_sync = model.no_sync

    loss = 0
    grad_norm = None
    if distributed_backend == DistributedBackend.torch:
        optimizer.zero_grad()

    with no_sync():
        for _ in range(gradient_accumulation_steps - 1):
            batch = get_next_batch(train_dataloader)
            with train_step_context:
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

    if distributed_backend == DistributedBackend.torch and hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(True)

    batch = get_next_batch(train_dataloader)
    with train_step_context:
        loss_micro_step = model(batch)
    loss += loss_micro_step

    # compute gradients
    if distributed_backend == DistributedBackend.deepspeed:
        model.backward(loss_micro_step)

        if gradient_clipping is not None:
            grad_norm = model.get_global_grad_norm()

        model.step()
    elif distributed_backend == DistributedBackend.torch:
        loss_micro_step.backward()

        if gradient_clipping is not None:
            assert ProcessGroupManager.get_tensor_parallel_world_size() == 1
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()
        lr_scheduler.step()
    else:
        raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    loss = loss / gradient_accumulation_steps
    loss = loss.item()
    grad_norm = grad_norm.item()

    return loss, grad_norm


def train(
    args: TrainingArgs,
    model: ModelWrapperForFinetuning,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: ResumableDataLoader,
    val_dataloader: ResumableDataLoader,
    experiments_tracker: ExperimentsTracker,
    starting_iteration: int = 0,
) -> None:
    """main training loop for the program

    Args:
        args (TrainingArgs): training args
        model (ModelWrapperForFinetuning): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LRScheduler): learning rate scheduler
        train_dataloader (ResumableDataLoader): training dataloader
        val_dataloader (ResumableDataLoader): validation dataloader
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

    loss_running_mean_tracker = RunningMean(window=args.logging_args.running_mean_window)

    model.train()

    # need this for iterating infinitely
    train_dataloader_infinite = infinite_iterator(train_dataloader)

    if eval_during_training:
        evaluate(val_dataloader, model, starting_iteration, experiments_tracker)

    train_step_context = contextlib.nullcontext()
    use_nvte_fp8 = (
        args.mixed_precision_args.dtype == "fp8" and args.mixed_precision_args.fp8_backend == FP8Backend.nvte
    )

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1

        if use_nvte_fp8:
            train_step_context = te.fp8_autocast(
                enabled=True,
                fp8_recipe=DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"),
            )

        loss_step, grad_norm_step = train_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            distributed_backend=distributed_backend,
            train_dataloader=train_dataloader_infinite,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            train_step_context=train_step_context,
        )

        if global_step % log_interval == 0:
            track_train_metrics(
                global_step=global_step,
                train_loss_step=loss_step,
                grad_norm_step=grad_norm_step,
                current_lr=(
                    model.lr_scheduler.get_lr()[0]
                    if distributed_backend == DistributedBackend.deepspeed
                    else lr_scheduler.get_lr()[0]
                ),
                experiments_tracker=experiments_tracker,
                loss_running_mean_tracker=loss_running_mean_tracker,
            )

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(val_dataloader, model, global_step, experiments_tracker)

        if global_step % save_interval == 0 or global_step == num_training_steps:
            save_checkpoint(args, model, optimizer, lr_scheduler, train_dataloader, experiments_tracker, global_step)


@torch.no_grad()
def evaluate(
    val_dataloader: ResumableDataLoader,
    model: ModelWrapperForFinetuning,
    global_step: int,
    experiments_tracker: ExperimentsTracker,
) -> float:
    """main validation loop for the program

    Args:
        val_dataloader (ResumableDataLoader): validation dataloader
        model (ModelWrapperForFinetuning): model
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker

    Returns:
        float: loss at the current step
    """

    if ProcessGroupManager.get_tensor_parallel_world_size() > 1:
        # other tensor parallel ranks need to be told if val dataloader is None or not
        is_val_dataloader_none = (
            val_dataloader is None or len(val_dataloader) == 0
            if ProcessGroupManager.get_tensor_parallel_rank() == 0
            else None
        )
        is_val_dataloader_none = Communication.broadcast_object(
            is_val_dataloader_none,
            src=ProcessGroupManager.get_tensor_parallel_first_rank(),
            group=ProcessGroupManager.get_tensor_parallel_group(),
        )
    else:
        is_val_dataloader_none = val_dataloader is None or len(val_dataloader) == 0

    if is_val_dataloader_none:
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
    init_distributed(
        tensor_parallel_size=args.distributed_args.tensor_parallel_size,
        data_parallel_size=args.distributed_args.data_parallel_size,
        data_parallel_replication_world_size=args.distributed_args.zero_topology.data_parallel_replication_world_size,
        data_parallel_sharding_world_size=args.distributed_args.zero_topology.data_parallel_sharding_world_size,
        timeout_minutes=args.distributed_args.timeout_minutes,
    )
    set_seed(args.random_args.seed)

    model = get_model(args, mode)

    train_dataloader = get_dataloader(
        args,
        split=DatasetSplit.train,
        mode=mode,
        tokenizer=model.tokenizer,
        is_encoder_decoder=model.is_encoder_decoder,
    )

    val_dataloader = None
    if args.training_parameters.eval_during_training:
        val_dataloader = get_dataloader(
            args,
            split=DatasetSplit.val,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
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
