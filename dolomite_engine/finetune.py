import logging
from contextlib import nullcontext
from functools import partial

import torch
from torch.distributed import ReduceOp
from torch.distributed.tensor.parallel import loss_parallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training, save_checkpoint
from .communication import Communication
from .data import ResumableDataLoader, get_dataloader, infinite_iterator
from .distributed import set_deepspeed_config, wrap_model_for_distributed_training
from .enums import DatasetSplit, DistributedBackend, FP8Backend, Mode
from .model_wrapper import ModelWrapperForFinetuning, get_model, log_model
from .optimization import get_optimizer, get_scheduler
from .train_utils import get_torch_profiler, track_train_metrics, train_step
from .utils import (
    ExperimentsTracker,
    ProcessGroupManager,
    init_distributed,
    is_transformer_engine_available,
    log_rank_0,
    setup_tf32,
)


if is_transformer_engine_available():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format


def track_val_metrics(global_step: int, val_loss: float, experiments_tracker: ExperimentsTracker) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        val_loss (float): validation loss for the validation data
        experiments_tracker (ExperimentsTracker): metrics tracker
    """

    log_rank_0(logging.INFO, f"step = {global_step}, val_loss = {val_loss:.4f}")
    experiments_tracker.track({"loss": val_loss}, step=global_step, context="val")


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

    model.train()

    # need this for iterating infinitely
    train_dataloader_infinite = infinite_iterator(train_dataloader)

    if eval_during_training:
        evaluate(val_dataloader, model, starting_iteration, experiments_tracker)

    forward_context = (
        partial(
            te.fp8_autocast,
            enabled=True,
            fp8_recipe=DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"),
        )
        if args.mixed_precision_args.dtype == "fp8" and args.mixed_precision_args.fp8_backend == FP8Backend.nvte
        else nullcontext
    )

    backward_context = loss_parallel if args.distributed_args.tensor_parallel_word_embeddings else nullcontext

    torch_profiler = get_torch_profiler(args.logging_args.torch_profiler_trace_path)

    if torch_profiler is not None:
        torch_profiler.__enter__()

    loss_running_sum = 0

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1

        loss_step, grad_norm_step = train_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            distributed_backend=distributed_backend,
            train_dataloader=train_dataloader_infinite,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            forward_context=forward_context,
            backward_context=backward_context,
        )

        loss_running_sum += loss_step

        if torch_profiler is not None:
            torch_profiler.step()

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
                loss_running_mean=loss_running_sum / log_interval,
            )

            loss_running_sum = 0

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(val_dataloader, model, global_step, experiments_tracker)

        if global_step % save_interval == 0 or global_step == num_training_steps:
            save_checkpoint(args, model, optimizer, lr_scheduler, train_dataloader, experiments_tracker, global_step)

    if torch_profiler is not None:
        torch_profiler.__exit__()


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
    micro_steps = 0

    for batch in val_dataloader:
        loss = model(batch)
        loss_sum += loss
        micro_steps += 1

    loss_mean = loss_sum / micro_steps
    torch.distributed.all_reduce(loss_mean, op=ReduceOp.AVG, group=ProcessGroupManager.get_data_parallel_group())
    loss_mean = loss_mean.item()

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

    if args.distributed_args.distributed_backend == DistributedBackend.deepspeed:
        set_deepspeed_config(args)

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

    model = wrap_model_for_distributed_training(args, model)

    if args.distributed_args.distributed_backend == DistributedBackend.torch:
        optimizer = get_optimizer(
            optimizer_class_name=args.optimizer_args.class_name,
            optimizer_class_args=args.optimizer_args.class_args,
            cpu_offload=args.distributed_args.cpu_offload,
            model=model,
            params_group_method=args.optimizer_args.params_group_method,
        )

        lr_scheduler = get_scheduler(
            optimizer=optimizer,
            num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
            num_constant_steps=args.lr_scheduler_args.num_constant_steps,
            num_decay_steps=args.lr_scheduler_args.num_decay_steps,
            num_training_steps=args.training_parameters.num_training_steps,
            lr_decay_style=args.lr_scheduler_args.lr_decay_style,
            lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
            extra_lr_scheduler_args=args.lr_scheduler_args.extra_lr_scheduler_args,
        )
    else:
        optimizer = None
        lr_scheduler = None

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

    ProcessGroupManager.destroy_process_groups()


if __name__ == "__main__":
    main()
