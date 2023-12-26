from typing import List, Tuple

import torch
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from transformers import set_seed

from engine.arguments import TrainingArgs, get_args
from engine.checkpointing import load_checkpoint_for_training, save_deepspeed_checkpoint
from engine.constants import DatasetSplit, Mode
from engine.data import ConcatenatedDatasets
from engine.model import Model
from engine.optimization import get_optimizer, get_scheduler_method
from engine.utils import (
    ExperimentsTracker,
    ProgressBar,
    RunningMean,
    deepspeed_initialize,
    init_distributed,
    print_rank_0,
    register_profiler,
    register_timer,
    setup_debugging,
    setup_tf32,
)


def track_train_metrics(
    global_step: int,
    train_loss_step: float,
    current_lr: float,
    experiments_tracker: ExperimentsTracker,
    loss_running_mean_tracker: RunningMean,
    progress_bar: ProgressBar,
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        train_loss_step (float): training loss at the current step
        current_lr (float): learning rate at the current step
        experiments_tracker (ExperimentsTracker): metrics tracker
        loss_running_mean_tracker (RunningMean): running mean accumulator for loss
        progress_bar (ProgressBar): progress bar for tracking training progress
    """

    # update loss running mean
    loss_running_mean = loss_running_mean_tracker.add_loss(train_loss_step)

    # track loss
    experiments_tracker.track(
        value=train_loss_step, name="loss", step=global_step, context={"subset": "train", "type": "step"}
    )
    experiments_tracker.track(
        value=loss_running_mean,
        name="loss",
        step=global_step,
        context={"subset": "train", "type": "running_mean"},
    )

    # track learning_rate
    experiments_tracker.track(value=current_lr, name="learning_rate", step=global_step)
    experiments_tracker.info(
        f"step = {global_step}, train_loss (batch) = {train_loss_step}, train_loss (running_mean) = {loss_running_mean}, learning_rate = {current_lr}"
    )

    # update metrics in progress bar
    progress_bar.track(loss_step=train_loss_step, loss_running_mean=loss_running_mean, current_lr=current_lr)


def track_val_metrics(global_step: int, val_loss: float, experiments_tracker: ExperimentsTracker) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        val_loss (float): validation loss for the validation data
        experiments_tracker (ExperimentsTracker): metrics tracker
    """

    print_rank_0(f"step = {global_step}, val_loss = {val_loss}")
    experiments_tracker.info(f"step = {global_step}, val_loss = {val_loss}")
    experiments_tracker.track(value=val_loss, name="loss", step=global_step, context={"subset": "val"})


@register_profiler("train_step")
@register_timer("train_step")
def train_step(model: DeepSpeedEngine, batch: Tuple[List[int]]) -> float:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model (DeepSpeedEngine): DeepSpeed sharded model
        batch (Tuple[List[int]]): a batch of examples on each GPU

    Returns:
        float: loss at the current step
    """

    loss = model(batch)

    # compute gradients
    model.backward(loss)
    # update weights and optimizer states
    model.step()

    loss_value = loss.item()
    return loss_value


def train(
    args: TrainingArgs,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: DeepSpeedEngine,
    experiments_tracker: ExperimentsTracker,
) -> None:
    """main training loop for the program

    Args:
        args (TrainingArgs): training args
        train_dataloader (DataLoader): training dataloader
        val_dataloader (DataLoader): validation dataloader
        model (DeepSpeedEngine): DeepSpeed sharded model
        experiments_tracker (ExperimentsTracker): metrics tracker
    """

    loss_running_mean_tracker = RunningMean()
    progress_bar = ProgressBar(0, args.num_training_steps)
    micro_step = 0

    model.train()
    train_loss_step_accumulator = 0

    # to run on multiple epochs
    while micro_step < args.num_training_steps * args.gradient_accumulation_steps:
        for batch in train_dataloader:
            # this completes the job
            if micro_step == args.num_training_steps * args.gradient_accumulation_steps:
                break

            if args.eval_during_training and micro_step % (args.eval_interval * args.gradient_accumulation_steps) == 0:
                val_loss = evaluate(val_dataloader, model)
                track_val_metrics(micro_step // args.gradient_accumulation_steps, val_loss, experiments_tracker)

            if micro_step != 0 and micro_step % (args.save_interval * args.gradient_accumulation_steps) == 0:
                save_deepspeed_checkpoint(model, args, args.save_path)

            train_loss_step_accumulator += train_step(model, batch)
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                track_train_metrics(
                    global_step=micro_step // args.gradient_accumulation_steps,
                    train_loss_step=train_loss_step_accumulator / args.gradient_accumulation_steps,
                    current_lr=model.lr_scheduler.get_lr()[0],
                    experiments_tracker=experiments_tracker,
                    loss_running_mean_tracker=loss_running_mean_tracker,
                    progress_bar=progress_bar,
                )

                train_loss_step_accumulator = 0
                progress_bar.update()

    if args.eval_during_training:
        val_loss = evaluate(val_dataloader, model)
        track_val_metrics(micro_step // args.gradient_accumulation_steps, val_loss, experiments_tracker)

    save_deepspeed_checkpoint(model, args, args.save_path)


@register_profiler("evaluate_dataset")
@torch.no_grad()
def evaluate(val_dataloader: DataLoader, model: Model) -> float:
    """main validation loop for the program

    Args:
        val_dataloader (DataLoader): validation dataloader
        model (DeepSpeedEngine): DeepSpeed sharded model

    Returns:
        float: loss at the current step
    """

    if val_dataloader is None:
        return

    model.eval()

    loss_sum = 0
    micro_step = 0
    progress_bar = ProgressBar(0, len(val_dataloader))

    for batch in val_dataloader:
        loss_value = model(batch).item()
        loss_sum += loss_value
        micro_step += 1
        progress_bar.update()

    loss_mean = loss_sum / micro_step

    model.train()

    return loss_mean


def main() -> None:
    """main program"""

    mode = Mode.training

    setup_tf32()

    args: TrainingArgs = get_args(mode)

    # initialize distributed with nccl for multi-node communications
    init_distributed()
    set_seed(args.seed)

    # setup deepspeed model
    model = Model(args, mode)

    # non-sharded training dataset
    train_dataset = ConcatenatedDatasets(
        args,
        split=DatasetSplit.train,
        mode=mode,
        tokenizer=model.tokenizer,
        is_encoder_decoder=model.is_encoder_decoder,
    )

    # non-sharded validation dataset
    val_dataset = None
    if args.eval_during_training:
        val_dataset = ConcatenatedDatasets(
            args,
            split=DatasetSplit.val,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
        )

    model.post_init()

    # setup Adam optimizer
    optimizer = get_optimizer(args, model.parameters())

    # setup learning rate schedule
    lr_scheduler = get_scheduler_method(args.lr_schedule)(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_training_steps
    )

    # shard the model and the optimizer
    model, (train_dataloader, val_dataloader) = deepspeed_initialize(
        args, model, optimizer, lr_scheduler, [train_dataset, val_dataset]
    )

    if args.load_path is not None:
        load_checkpoint_for_training(model, args.load_path)

    experiments_tracker = ExperimentsTracker(__name__, args.experiment_name, args.aim_repo, args.logdir)
    # track all hyperparams in args
    experiments_tracker.log_args(args)

    # main training loop
    train(args, train_dataloader, val_dataloader, model, experiments_tracker)


if __name__ == "__main__":
    main()
