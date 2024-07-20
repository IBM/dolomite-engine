import logging
import time
from contextlib import nullcontext
from functools import partial

import torch
from torch.distributed.tensor.parallel import loss_parallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training, save_checkpoint
from .communication import Communication
from .data import get_megatron_gpt_dataloaders
from .distributed import set_deepspeed_config, wrap_model_for_distributed_training
from .enums import DistributedBackend, FP8Backend, Mode
from .model_wrapper import ModelWrapperForPretraining, get_model, log_model
from .train_utils import get_model_tflops, get_torch_profiler, track_train_metrics, train_step
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


def track_val_metrics(
    global_step: int, val_loss: float, experiments_tracker: ExperimentsTracker, group_name: str | None = None
) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        val_loss (float): validation loss for the validation data
        experiments_tracker (ExperimentsTracker): metrics tracker
        group_name (str | None): group name for the validation / test set
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
    val_dataloaders: list[DataLoader],
    test_dataloaders: list[DataLoader],
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
        val_dataloaders (list[DataLoader]): validation dataloaders
        test_dataloaders (list[DataLoader]): test dataloaders
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

    loss_running_mean_tracker = RunningMean(window=args.logging_args.running_mean_window)

    model.train()

    if eval_during_training:
        eval_steps = args.datasets[0].class_args.get("eval_steps")
        evaluate(val_dataloaders, model, starting_iteration, experiments_tracker, eval_steps, group_names)

    micro_batch_size = args.training_parameters.micro_batch_size
    sequence_length = args.datasets[0].class_args.get("sequence_length")
    global_batch_size = (
        micro_batch_size * gradient_accumulation_steps * ProcessGroupManager.get_data_parallel_world_size()
    )
    tokens_per_batch = global_batch_size * sequence_length

    # model flops per GPU
    model_flops = (
        get_model_tflops(
            model_class=args.model_args.model_class,
            config=model.config,
            batch_size=global_batch_size,
            sequence_length=sequence_length,
            gradient_checkpointing_method=args.distributed_args.gradient_checkpointing_method,
            gradient_checkpointing_args=args.distributed_args.gradient_checkpointing_args,
        )
        / ProcessGroupManager.get_world_size()
    )

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
            forward_context=forward_context,
            backward_context=backward_context,
        )

        if torch_profiler is not None:
            torch_profiler.step()

        if global_step % log_interval == 0:
            time_elapsed = time.perf_counter() - start_time
            step_time = time_elapsed / steps_since_start_time

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
                {
                    "consumed_samples": global_step
                    * micro_batch_size
                    * gradient_accumulation_steps
                    * ProcessGroupManager.get_data_parallel_world_size()
                },
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0

    if eval_during_training:
        evaluate(test_dataloaders, model, global_step, experiments_tracker, eval_steps, group_names)

    if torch_profiler is not None:
        torch_profiler.__exit__()


@torch.no_grad()
def evaluate(
    val_dataloaders: list[DataLoader],
    model: ModelWrapperForPretraining,
    global_step: int,
    experiments_tracker: ExperimentsTracker,
    eval_steps: int,
    group_names: list[str],
) -> float:
    """main validation loop for the program

    Args:
        val_dataloaders (list[DataLoader]): list of validation dataloaders
        model (ModelWrapperForPretraining): model
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        eval_steps (int): number of steps to run eval for
        group_names (list[str]): names of the datasets in validation/test group

    Returns:
        float: loss at the current step
    """

    if ProcessGroupManager.get_tensor_parallel_world_size() > 1:
        # other tensor parallel ranks need to be told if val dataloader is None or not
        is_val_dataloader_none = (
            val_dataloaders is None or len(val_dataloaders) == 0
            if ProcessGroupManager.get_tensor_parallel_rank() == 0
            else None
        )
        is_val_dataloader_none = Communication.broadcast_object(
            is_val_dataloader_none,
            src=ProcessGroupManager.get_tensor_parallel_first_rank(),
            group=ProcessGroupManager.get_tensor_parallel_group(),
        )
    else:
        is_val_dataloader_none = val_dataloaders is None or len(val_dataloaders) == 0

    if is_val_dataloader_none:
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
    model, optimizer, lr_scheduler = wrap_model_for_distributed_training(args, model)

    log_model(model)

    starting_iteration = 0
    metadata = None
    experiments_tracker_state_dict = None
    if args.load_args is not None:
        starting_iteration, metadata, experiments_tracker_state_dict = load_checkpoint_for_training(
            args, model, optimizer, lr_scheduler, None
        )

        # metadata field contains the dataloader state so we need to reset it here
        if not args.load_args.load_dataloader_state and metadata is not None:
            metadata["consumed_samples"] = 0

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

    ProcessGroupManager.destroy_process_groups()


if __name__ == "__main__":
    main()
