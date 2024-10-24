from contextlib import nullcontext
from functools import partial

import torch
from torch.distributed._tensor.api import DTensor
from torch.distributed.tensor.parallel import loss_parallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training, save_checkpoint
from .data import ResumableDataLoader, custom_iterator, get_dataloader, get_next_batch
from .distributed import wrap_model_for_distributed_training
from .enums import DatasetSplit, FP8Backend, Mode, TuningMethod
from .model_wrapper import ModelWrapperForFinetuning, get_model, log_model
from .optimization import get_optimizer, get_scheduler, log_optimizer
from .train_utils import all_reduce_metrics_tracker, get_torch_profiler, track_metrics, train_step
from .utils import (
    ExperimentsTracker,
    MetricsTrackingDict,
    ProcessGroupManager,
    init_distributed,
    is_transformer_engine_available,
    setup_tf32,
)


if is_transformer_engine_available():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format


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
    save_interval = args.save_args.save_interval
    log_interval = args.logging_args.log_interval

    model.train()

    # need this for iterating infinitely
    train_dataloader_infinite = custom_iterator(train_dataloader, infinite=True)

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

    metrics_tracker = MetricsTrackingDict({})

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1

        loss_step_dict = train_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader_infinite,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            forward_context=forward_context,
            backward_context=backward_context,
            sync_every_gradient_accumulation_step=args.distributed_args.sync_every_gradient_accumulation_step,
        )

        metrics_tracker = metrics_tracker + loss_step_dict

        if torch_profiler is not None:
            torch_profiler.step()

        if global_step % log_interval == 0:
            metrics_tracker = metrics_tracker / log_interval
            metrics_tracker["learning_rate"] = lr_scheduler.get_lr()[0]

            track_metrics(
                global_step=global_step,
                experiments_tracker=experiments_tracker,
                metrics_tracker=metrics_tracker,
                context="train",
            )

            metrics_tracker = MetricsTrackingDict({})

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
) -> MetricsTrackingDict:
    """main validation loop for the program

    Args:
        val_dataloader (ResumableDataLoader): validation dataloader
        model (ModelWrapperForFinetuning): model
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker

    Returns:
        MetricsTrackingDict: metrics tracker
    """

    if ProcessGroupManager.is_tensor_parallel_enabled():
        if ProcessGroupManager.get_tensor_parallel_rank() == 0:
            num_steps = 0 if val_dataloader is None else len(val_dataloader)
        else:
            num_steps = 0

        num_steps = torch.tensor(num_steps, device=torch.cuda.current_device(), dtype=torch.long)
        torch.distributed.all_reduce(num_steps, group=ProcessGroupManager.get_tensor_parallel_group())
        num_steps = num_steps.item()
    else:
        num_steps = 0 if val_dataloader is None else len(val_dataloader)

    if num_steps == 0:
        return

    model.eval()

    metrics_tracker = MetricsTrackingDict({})
    val_dataloader = custom_iterator(val_dataloader, infinite=False)

    for _ in range(num_steps):
        batch = get_next_batch(val_dataloader)
        loss_step_dict = model(batch)
        metrics_tracker = metrics_tracker + loss_step_dict

    metrics_tracker = metrics_tracker / num_steps

    for key in metrics_tracker:
        if isinstance(metrics_tracker[key], DTensor):
            metrics_tracker[key] = metrics_tracker[key].to_local()

    metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

    track_metrics(
        global_step=global_step,
        experiments_tracker=experiments_tracker,
        metrics_tracker=metrics_tracker,
        context="val",
    )

    model.train()

    return metrics_tracker


def main() -> None:
    """main program"""

    mode = Mode.training

    setup_tf32()

    args: TrainingArgs = get_args(mode)

    assert args.tuning_args.tuning_method in [
        TuningMethod.full_finetuning,
        TuningMethod.lora,
        TuningMethod.prompt_tuning,
    ], f"unexpected tuning method ({args.tuning_args.tuning_method})"

    # initialize distributed with nccl for multi-node communications
    init_distributed(
        tensor_parallel_size=args.distributed_args.tensor_parallel_size,
        data_parallel_size=args.distributed_args.data_parallel_size,
        data_parallel_replication_world_size=args.distributed_args.zero_topology.data_parallel_replication_world_size,
        data_parallel_sharding_world_size=args.distributed_args.zero_topology.data_parallel_sharding_world_size,
        zero_stage=args.distributed_args.stage,
        timeout_minutes=args.distributed_args.timeout_minutes,
        use_async_tensor_parallel=args.distributed_args.use_async_tensor_parallel,
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

    model = wrap_model_for_distributed_training(args, model)

    optimizer = get_optimizer(
        optimizer_class_name=args.optimizer_args.class_name,
        optimizer_class_args=args.optimizer_args.class_args,
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

    log_model(model)
    log_optimizer(optimizer)

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
