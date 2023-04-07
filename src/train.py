from argparse import Namespace
from typing import List, Tuple

import torch
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from transformers import set_seed

from src.arguments import get_args
from src.constants import DatasetSplit, Mode
from src.data import ConcatenatedDatasets
from src.model import Model, ModelCheckpointer
from src.optimization import get_optimizer, get_scheduler_method
from src.utils import (
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
    args: Namespace,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: DeepSpeedEngine,
    experiments_tracker: ExperimentsTracker,
) -> None:
    """main training loop for the program

    Args:
        args (Namespace): training args
        train_dataloader (DataLoader): training dataloader
        val_dataloader (DataLoader): validation dataloader
        model (DeepSpeedEngine): DeepSpeed sharded model
        experiments_tracker (ExperimentsTracker): metrics tracker
    """

    loss_running_mean_tracker = RunningMean()
    progress_bar = ProgressBar(0, args.num_training_steps)
    global_step = 0

    model.train()

    # to run on multiple epochs
    while global_step < args.num_training_steps:
        for batch in train_dataloader:
            # this completes the job
            if global_step == args.num_training_steps:
                break

            if global_step % args.eval_and_save_interval == 0:
                if not args.no_eval:
                    val_loss = evaluate(val_dataloader, model)
                    track_val_metrics(global_step, val_loss, experiments_tracker)

                ModelCheckpointer.save_checkpoint(model, args.save_path)

            train_loss_step = train_step(model, batch)

            track_train_metrics(
                global_step=global_step,
                train_loss_step=train_loss_step,
                current_lr=model.lr_scheduler.get_lr()[0],
                experiments_tracker=experiments_tracker,
                loss_running_mean_tracker=loss_running_mean_tracker,
                progress_bar=progress_bar,
            )

            global_step += 1
            progress_bar.update()

    if not args.no_eval:
        val_loss = evaluate(val_dataloader, model)
        track_val_metrics(global_step, val_loss, experiments_tracker)

    ModelCheckpointer.save_checkpoint(model, args.save_path)


@register_profiler("evaluate_dataset")
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
    global_step = 0
    progress_bar = ProgressBar(0, len(val_dataloader))

    with torch.inference_mode():
        for batch in val_dataloader:
            loss_value = model(batch).item()
            loss_sum += loss_value
            global_step += 1
            progress_bar.update()

    loss_mean = loss_sum / global_step

    model.train()

    return loss_mean


def main() -> None:
    """main program"""

    mode = Mode.training

    setup_tf32()

    args = get_args(mode)

    setup_debugging()

    # initialize distributed with nccl for multi-node communications
    init_distributed("nccl")
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
    if not args.no_eval:
        val_dataset = ConcatenatedDatasets(
            args,
            split=DatasetSplit.val,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
        )

    model.post_init()

    # setup Adam optimizer
    optimizer = get_optimizer(
        args.cpu_offload,
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # setup learning rate schedule
    lr_scheduler = get_scheduler_method(args.lr_schedule)(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_training_steps
    )

    # shard the model and the optimizer
    model, (train_dataloader, val_dataloader) = deepspeed_initialize(
        args, model, optimizer, lr_scheduler, [train_dataset, val_dataset]
    )

    experiments_tracker = ExperimentsTracker(__name__, args.experiment_name, args.aim_repo, args.logdir)
    # track all hyperparams in args
    experiments_tracker.log_args(args)

    # main training loop
    train(args, train_dataloader, val_dataloader, model, experiments_tracker)


if __name__ == "__main__":
    main()
