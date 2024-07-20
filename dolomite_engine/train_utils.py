import logging
from contextlib import AbstractContextManager, nullcontext

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .data import ResumableDataLoader, get_next_batch
from .enums import DistributedBackend, GradientCheckpointingMethod
from .hf_models import is_custom_model
from .hf_models.modeling_utils import is_glu
from .model_wrapper import ModelWrapperForFinetuning
from .utils import ExperimentsTracker, ProcessGroupManager, RunningMean, log_rank_0


def train_step(
    model: ModelWrapperForFinetuning,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    distributed_backend: DistributedBackend,
    train_dataloader: ResumableDataLoader,
    gradient_accumulation_steps: int,
    gradient_clipping: float,
    forward_context: AbstractContextManager,
    backward_context: AbstractContextManager,
) -> tuple[float, float]:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model (ModelWrapperForFinetuning): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LamdaLR): learning rate scheduler
        distributed_backend (DistributedBackend): distributed backend
        train_dataloader (ResumableDataLoader): training dataloader
        gradient_accumulation_steps (int): gradient accumulation steps
        gradient_clipping (float): gradient clipping value
        forward_context (AbstractContextManager): a context that is used for every model forward call
        backward_context (AbstractContextManager): a context that is used for every model backward call

    Returns:
        tuple[float, float]: loss at the current step, grad norm at the current step
    """

    no_sync = nullcontext
    if distributed_backend == DistributedBackend.torch:
        fsdp_algorithm = 2 if hasattr(model, "set_requires_gradient_sync") else 1

        if fsdp_algorithm == 1:
            no_sync = model.no_sync
        else:
            model.set_requires_gradient_sync(False)

    loss = 0
    grad_norm = None
    if distributed_backend == DistributedBackend.torch:
        optimizer.zero_grad()

    with no_sync():
        for _ in range(gradient_accumulation_steps - 1):
            batch = get_next_batch(train_dataloader)
            with forward_context():
                loss_micro_step = model(batch)
            loss += loss_micro_step

            # compute gradients
            if distributed_backend == DistributedBackend.deepspeed:
                with backward_context():
                    model.backward(loss_micro_step)
                model.step()
            elif distributed_backend == DistributedBackend.torch:
                with backward_context():
                    loss_micro_step.backward()
            else:
                raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    if distributed_backend == DistributedBackend.torch and fsdp_algorithm == 2:
        model.set_requires_gradient_sync(True)

    batch = get_next_batch(train_dataloader)
    with forward_context():
        loss_micro_step = model(batch)
    loss += loss_micro_step

    # compute gradients
    if distributed_backend == DistributedBackend.deepspeed:
        with backward_context():
            model.backward(loss_micro_step)

        if gradient_clipping is not None:
            grad_norm = model.get_global_grad_norm()

        model.step()
    elif distributed_backend == DistributedBackend.torch:
        with backward_context():
            loss_micro_step.backward()

        if gradient_clipping is not None:
            if fsdp_algorithm == 1:
                grad_norm = model.clip_grad_norm_(gradient_clipping)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()
        lr_scheduler.step()
    else:
        raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    loss = loss / gradient_accumulation_steps
    loss = loss.item()
    grad_norm = 0 if grad_norm is None else grad_norm.item()

    return loss, grad_norm


def track_train_metrics(
    global_step: int,
    train_loss_step: float,
    grad_norm_step: float,
    current_lr: float,
    experiments_tracker: ExperimentsTracker,
    loss_running_mean_tracker: RunningMean,
    flops: float | None = None,
    billion_tokens_per_day: float | None = None,
    step_time: float | None = None,
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        train_loss_step (float): training loss at the current step
        current_lr (float): learning rate at the current step
        experiments_tracker (ExperimentsTracker): metrics tracker
        loss_running_mean_tracker (RunningMean): running mean accumulator for loss
        flops (float | None, optional): total model flops. Defaults to None
        billion_tokens_per_day (float | None, optional): billions of tokens per day. Defaults to None
        step_time (float | None, optional): time per step in seconds
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


def get_torch_profiler(torch_profiler_trace_path: str) -> torch.profiler.profile:
    torch_profiler = None
    if torch_profiler_trace_path is not None:
        torch_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=5 if ProcessGroupManager.get_global_rank() == 0 else 150000, warmup=5, active=1, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(torch_profiler_trace_path),
            record_shapes=True,
        )

    return torch_profiler


def get_model_tflops(
    model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
    config: AutoConfig,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing_method: GradientCheckpointingMethod | None,
    gradient_checkpointing_args: dict,
) -> None:
    if not is_custom_model(model_class, config.model_type):
        return 0

    b = batch_size
    s = sequence_length
    h = config.n_embd
    f = config.n_inner
    n = config.n_head
    k = config.num_key_value_heads
    l = config.n_layer
    v = config.vocab_size

    mlp_flops = 4 * b * s * h * f
    if is_glu(config.activation_function):
        mlp_flops += 2 * b * s * h * f

    attention_flops = 4 * b * s * h * (h * (1 + k / n) + s)

    forward_flops = attention_flops + mlp_flops

    if gradient_checkpointing_method == GradientCheckpointingMethod.block:
        num_layers_checkpointed = l // gradient_checkpointing_args.get("checkpoint_every", 1)
        fraction_of_layers_checkpointed = num_layers_checkpointed / l
        backward_flops = (2 + fraction_of_layers_checkpointed) * forward_flops
    else:
        backward_flops = 2 * forward_flops

    model_flops = l * (forward_flops + backward_flops)
    model_flops += 6 * b * s * h * v
    model_flops /= 10**12

    return model_flops
