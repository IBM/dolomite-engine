import logging
import time
from contextlib import nullcontext
from functools import partial

import torch
from torch.distributed._tensor.api import DTensor
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.data import DataLoader
from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training, save_checkpoint
from .communication import Communication
from .containers import LRSchedulerContainer, ModelContainer, OptimizerContainer, log_model_optimizer_container
from .data import get_megatron_gpt_dataloaders, get_next_batch
from .distributed import wrap_model_container_for_distributed_training
from .enums import FP8Backend, Mode, TuningMethod
from .model_wrapper import get_model_container
from .optimization import get_optimizer_container, get_scheduler_container
from .train_utils import all_reduce_metrics_tracker, get_model_tflops, get_torch_profiler, track_metrics, train_step
from .utils import (
    ExperimentsTracker,
    MetricsTrackingDict,
    ProcessGroupManager,
    init_distributed,
    is_transformer_engine_available,
    log_rank_0,
    setup_tf32,
)


if is_transformer_engine_available():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format


def track_val_metrics(
    global_step: int,
    experiments_tracker: ExperimentsTracker,
    metrics_tracker: MetricsTrackingDict,
    group_name: str | None = None,
) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): experiments tracker
        metrics_tracker (MetricsTrackingDict): metrics tracker
        group_name (str | None): group name for the validation / test set
    """

    context = "val"

    message = f"step = {global_step}"
    if group_name is not None:
        message += f", group_name = {group_name}"

    for key in metrics_tracker:
        message += f", {context}-{key} = {metrics_tracker[key]:.4f}"

    log_rank_0(logging.INFO, message)

    if group_name is None:
        message = metrics_tracker.get_dict()
    else:
        message = {}
        for key in metrics_tracker:
            message[f"{group_name}-{key}"] = metrics_tracker[key]

    experiments_tracker.track(message, step=global_step, context=context)


def train(
    args: TrainingArgs,
    model_container: ModelContainer,
    pipeline_schedule: _PipelineSchedule,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: DataLoader,
    val_dataloaders: list[DataLoader],
    test_dataloaders: list[DataLoader],
    experiments_tracker: ExperimentsTracker,
    starting_iteration: int = 0,
) -> None:
    """main training loop for the program

    Args:
        args (TrainingArgs): training args
        model_container (ModelContainer): container of models
        pipeline_schedule (_PipelineSchedule): pipeline schedule
        optimizer_container (OptimizerContainer): container of optimizers
        lr_scheduler_container (LRSchedulerContainer): container of learning rate schedulers
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
    save_interval = args.save_args.save_interval
    log_interval = args.logging_args.log_interval

    val_weighted_split_paths = args.datasets[0].class_args.get("val_weighted_split_paths")
    group_names = [None]
    if val_weighted_split_paths is not None:
        group_names = [key for key in val_weighted_split_paths.keys()[0]]

    model_container.train()

    if eval_during_training:
        eval_steps = args.datasets[0].class_args.get("eval_steps")
        evaluate(val_dataloaders, model_container, starting_iteration, experiments_tracker, eval_steps, group_names)

    micro_batch_size = args.training_parameters.micro_batch_size
    sequence_length = args.datasets[0].class_args.get("sequence_length")
    local_batch_size = micro_batch_size * gradient_accumulation_steps
    global_batch_size = local_batch_size * ProcessGroupManager.get_data_parallel_world_size()
    tokens_per_batch = global_batch_size * sequence_length

    dp_world_size = ProcessGroupManager.get_data_parallel_world_size()

    # model flops per GPU
    model_flops = (
        get_model_tflops(
            model_class=args.model_args.model_class,
            config=model_container[0].config,
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
    metrics_tracker = MetricsTrackingDict({})

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1
        steps_since_start_time += 1

        loss_step_dict = train_step(
            model_container=model_container,
            pipeline_schedule=pipeline_schedule,
            optimizer_container=optimizer_container,
            lr_scheduler_container=lr_scheduler_container,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            forward_context=forward_context,
            backward_context=backward_context,
            sync_every_gradient_accumulation_step=args.distributed_args.sync_every_gradient_accumulation_step,
            is_pipeline_parallel_enabled=args.distributed_args.num_pipeline_stages > 1,
            batch_size=local_batch_size,
            sequence_length=sequence_length,
        )

        metrics_tracker = metrics_tracker + loss_step_dict

        if torch_profiler is not None:
            torch_profiler.step()

        if global_step % log_interval == 0:
            metrics_tracker = metrics_tracker / log_interval

            time_elapsed = time.perf_counter() - start_time
            step_time = time_elapsed / steps_since_start_time

            metrics_tracker["learning_rate"] = lr_scheduler_container[0].get_lr()[0]

            if model_flops is not None:
                metrics_tracker["FLOPs"] = model_flops * steps_since_start_time / time_elapsed

            metrics_tracker["billion_tokens_per_day"] = tokens_per_batch * 86400 / step_time / 1e9
            metrics_tracker["step_time (sec)"] = step_time

            track_metrics(
                global_step=global_step,
                experiments_tracker=experiments_tracker,
                metrics_tracker=metrics_tracker,
                context="train",
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0
            metrics_tracker = MetricsTrackingDict({})

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(val_dataloaders, model_container, global_step, experiments_tracker, eval_steps, group_names)

        if global_step % save_interval == 0 or global_step == num_training_steps:
            save_checkpoint(
                args=args,
                model_container=model_container,
                optimizer_container=optimizer_container,
                lr_scheduler_container=lr_scheduler_container,
                train_dataloader=None,
                experiments_tracker=experiments_tracker,
                iteration=global_step,
                metadata={
                    "consumed_samples": global_step * micro_batch_size * gradient_accumulation_steps * dp_world_size
                },
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0

    if eval_during_training:
        evaluate(test_dataloaders, model_container, global_step, experiments_tracker, eval_steps, group_names)

    if torch_profiler is not None:
        torch_profiler.__exit__()


@torch.no_grad()
def evaluate(
    val_dataloaders: list[DataLoader],
    model_container: ModelContainer,
    global_step: int,
    experiments_tracker: ExperimentsTracker,
    eval_steps: int,
    group_names: list[str],
) -> float:
    """main validation loop for the program

    Args:
        val_dataloaders (list[DataLoader]): list of validation dataloaders
        model_container (ModelContainer): container of models
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        eval_steps (int): number of steps to run eval for
        group_names (list[str]): names of the datasets in validation/test group

    Returns:
        MetricsTrackingDict: metrics tracker
    """

    assert len(model_container) == 1
    model = model_container[0]

    if ProcessGroupManager.is_tensor_parallel_enabled():
        # other tensor parallel ranks need to be told if val dataloader is None or not
        is_val_dataloader_none = (
            val_dataloaders is None or len(val_dataloaders) == 0
            if ProcessGroupManager.is_tensor_parallel_first_rank()
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
        metrics_tracker = MetricsTrackingDict({})

        for _ in range(eval_steps):
            batch = get_next_batch(val_dataloader)
            loss_step_dict = model(batch)
            metrics_tracker = metrics_tracker + loss_step_dict

        metrics_tracker = metrics_tracker / eval_steps

        for key in metrics_tracker:
            if isinstance(metrics_tracker[key], DTensor):
                metrics_tracker[key] = metrics_tracker[key].to_local()

        metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

        track_val_metrics(
            global_step=global_step,
            experiments_tracker=experiments_tracker,
            metrics_tracker=metrics_tracker,
            group_name=group_name,
        )

    model.train()

    return metrics_tracker


def main(mode: Mode = Mode.training) -> None:
    """main program"""

    setup_tf32()

    args: TrainingArgs = get_args(mode)

    if mode == Mode.training:
        assert (
            args.tuning_args.tuning_method == TuningMethod.pretraining
        ), f"unexpected tuning method ({args.tuning_args.tuning_method})"
    elif mode == Mode.distillation:
        assert args.distributed_args.fsdp_algorithm == 2, "Distillation is only supported with FSDP-2"

        assert (
            args.tuning_args.tuning_method == TuningMethod.distillation
        ), f"unexpected tuning method ({args.tuning_args.tuning_method})"

    # initialize distributed with nccl for multi-node communications
    init_distributed(
        tensor_parallel_world_size=args.distributed_args.tensor_parallel_world_size,
        pipeline_parallel_world_size=args.distributed_args.pipeline_parallel_world_size,
        data_parallel_size=args.distributed_args.data_parallel_size,
        data_parallel_replication_world_size=args.distributed_args.zero_topology.data_parallel_replication_world_size,
        data_parallel_sharding_world_size=args.distributed_args.zero_topology.data_parallel_sharding_world_size,
        zero_stage=args.distributed_args.stage,
        timeout_minutes=args.distributed_args.timeout_minutes,
        use_async_tensor_parallel=args.distributed_args.use_async_tensor_parallel,
    )

    set_seed(args.random_args.seed)

    if mode == Mode.distillation:
        assert args.distributed_args.num_pipeline_stages == 1, "pipeline parallel is not supported with distillation"

    model_container = get_model_container(args, mode)
    model_container, pipeline_schedule = wrap_model_container_for_distributed_training(args, model_container)

    optimizer_container = get_optimizer_container(
        optimizer_class_name=args.optimizer_args.class_name,
        optimizer_class_args=args.optimizer_args.class_args,
        model_container=model_container,
        params_group_method=args.optimizer_args.params_group_method,
    )

    lr_scheduler_container = get_scheduler_container(
        optimizer_container=optimizer_container,
        num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
        num_constant_steps=args.lr_scheduler_args.num_constant_steps,
        num_decay_steps=args.lr_scheduler_args.num_decay_steps,
        num_training_steps=args.training_parameters.num_training_steps,
        lr_decay_style=args.lr_scheduler_args.lr_decay_style,
        lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
        extra_lr_scheduler_args=args.lr_scheduler_args.extra_lr_scheduler_args,
    )

    log_model_optimizer_container(model_container, optimizer_container)

    starting_iteration = 0
    metadata = None
    experiments_tracker_state_dict = None
    if args.load_args is not None:
        starting_iteration, metadata, experiments_tracker_state_dict = load_checkpoint_for_training(
            args, model_container, optimizer_container, lr_scheduler_container, None
        )

        # metadata field contains the dataloader state so we need to reset it here
        if not args.load_args.load_dataloader_state and metadata is not None:
            metadata["consumed_samples"] = 0

    train_dataloader, val_dataloaders, test_dataloaders = get_megatron_gpt_dataloaders(
        args, model_container[0].tokenizer, 0 if metadata is None else metadata["consumed_samples"]
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
        model_container=model_container,
        pipeline_schedule=pipeline_schedule,
        optimizer_container=optimizer_container,
        lr_scheduler_container=lr_scheduler_container,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        test_dataloaders=test_dataloaders,
        experiments_tracker=experiments_tracker,
        starting_iteration=starting_iteration,
    )

    ProcessGroupManager.destroy_process_groups()


if __name__ == "__main__":
    main()
