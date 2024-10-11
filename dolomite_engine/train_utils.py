import logging
from contextlib import AbstractContextManager, nullcontext

import torch
from torch.distributed import ReduceOp
from torch.distributed._tensor.api import DTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .data import ResumableDataLoader, get_next_batch
from .enums import DistributedBackend, GradientCheckpointingMethod
from .hf_models import is_custom_model
from .hf_models.modeling_utils import is_glu
from .model_wrapper import ModelWrapperForFinetuning
from .utils import ExperimentsTracker, MetricsTrackingDict, ProcessGroupManager, log_rank_0


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
    sync_every_gradient_accumulation_step: bool,
) -> MetricsTrackingDict:
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
        sync_every_gradient_accumulation_step (bool): whether to sync on every gradient accumulation step

    Returns:
        MetricsTrackingDict: metrics to track
    """

    if distributed_backend == DistributedBackend.torch:
        fsdp_algorithm = 2 if hasattr(model, "set_requires_gradient_sync") else 1

    no_sync = nullcontext
    if not sync_every_gradient_accumulation_step:
        if fsdp_algorithm == 1:
            no_sync = model.no_sync
        else:
            model.set_requires_gradient_sync(False)

    metrics_tracker = MetricsTrackingDict({})
    grad_norm = None
    if distributed_backend == DistributedBackend.torch:
        optimizer.zero_grad()

    with no_sync():
        for _ in range(gradient_accumulation_steps - 1):
            batch = get_next_batch(train_dataloader)
            with forward_context():
                loss_micro_step_dict = model(batch)

            with torch.inference_mode():
                metrics_tracker = metrics_tracker + loss_micro_step_dict

            # compute gradients
            if distributed_backend == DistributedBackend.deepspeed:
                with backward_context():
                    model.backward(loss_micro_step_dict["loss"])
                model.step()
            elif distributed_backend == DistributedBackend.torch:
                with backward_context():
                    loss_micro_step_dict["loss"].backward()
            else:
                raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    if distributed_backend == DistributedBackend.torch and fsdp_algorithm == 2:
        model.set_requires_gradient_sync(True)

    batch = get_next_batch(train_dataloader)
    with forward_context():
        loss_micro_step_dict = model(batch)

    with torch.inference_mode():
        metrics_tracker = metrics_tracker + loss_micro_step_dict

    # compute gradients
    if distributed_backend == DistributedBackend.deepspeed:
        with backward_context():
            model.backward(loss_micro_step_dict["loss"])

        if gradient_clipping is not None:
            grad_norm = model.get_global_grad_norm()

        model.step()
    elif distributed_backend == DistributedBackend.torch:
        with backward_context():
            loss_micro_step_dict["loss"].backward()

        if gradient_clipping is not None:
            if fsdp_algorithm == 1:
                grad_norm = model.clip_grad_norm_(gradient_clipping)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()
        lr_scheduler.step()
    else:
        raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    with torch.inference_mode():
        metrics_tracker = metrics_tracker / gradient_accumulation_steps

        metrics_tracker["grad_norm"] = (
            torch.tensor(0, device=torch.cuda.current_device()) if grad_norm is None else grad_norm
        )

        for key in metrics_tracker:
            if isinstance(metrics_tracker[key], DTensor):
                metrics_tracker[key] = metrics_tracker[key].to_local()

        metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

    return metrics_tracker


def all_reduce_metrics_tracker(metrics_tracker: MetricsTrackingDict) -> MetricsTrackingDict:
    tensor = [metrics_tracker[key] for key in metrics_tracker]
    tensor = torch.stack(tensor) / ProcessGroupManager.get_data_parallel_world_size()
    tensor = tensor.cpu()
    # gloo op doesn't support averaging so we do sum and divide by world size above
    torch.distributed.all_reduce(tensor, group=ProcessGroupManager.get_data_parallel_group())
    tensor = tensor.tolist()

    for i, key in enumerate(metrics_tracker):
        metrics_tracker[key] = tensor[i]

    return metrics_tracker


def track_metrics(
    global_step: int, experiments_tracker: ExperimentsTracker, metrics_tracker: MetricsTrackingDict, context: str
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        metrics_tracker (float): metrics tracker
        context (str): experiment context
    """

    # experiments tracker
    experiments_tracker.track(metrics_tracker.get_dict(), step=global_step, context=context)

    message = f"step = {global_step}"
    for key in metrics_tracker:
        if key == "learning_rate":
            message += f", {key} = {metrics_tracker[key]:.4e}"
        else:
            message += f", {context}-{key} = {metrics_tracker[key]:.4f}"

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
        num_layers_checkpointed = gradient_checkpointing_args.get("num_blocks", l)
        fraction_of_layers_checkpointed = num_layers_checkpointed / l
        backward_flops = (2 + fraction_of_layers_checkpointed) * forward_flops
    else:
        backward_flops = 2 * forward_flops

    model_flops = l * (forward_flops + backward_flops)
    model_flops += 6 * b * s * h * v
    model_flops /= 10**12

    return model_flops
