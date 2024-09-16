from contextlib import AbstractContextManager, nullcontext

import torch
from torch.distributed import ReduceOp
from torch.distributed.tensor.parallel import loss_parallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .arguments import TrainingArgs
from .checkpointing import save_checkpoint
from .communication import Communication
from .data import ResumableDataLoader, get_next_batch
from .enums import DistributedBackend, FP8Backend, Mode
from .model_wrapper import ModelWrapperForFinetuning, ModelWrapperForPretraining
from .pretrain import main, track_val_metrics
from .pretrain import train as pretrain
from .train_utils import get_model_tflops, get_torch_profiler, track_metrics, train_step
from .utils import ExperimentsTracker, ProcessGroupManager, is_transformer_engine_available, log_rank_0


if is_transformer_engine_available():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format


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
    lm_loss = 0
    kl_divergence = 0
    grad_norm = None
    if distributed_backend == DistributedBackend.torch:
        optimizer.zero_grad()

    with no_sync():
        for _ in range(gradient_accumulation_steps - 1):
            batch = get_next_batch(train_dataloader)
            with forward_context():
                loss_micro_step, lm_loss_step, kl_divergence_step = model(batch)
            loss += loss_micro_step
            lm_loss += lm_loss_step
            kl_divergence += kl_divergence_step

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
        loss_micro_step, lm_loss_step, kl_divergence_step = model(batch)
    loss += loss_micro_step
    lm_loss += lm_loss_step
    kl_divergence += kl_divergence_step

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

    loss /= gradient_accumulation_steps
    lm_loss /= gradient_accumulation_steps
    kl_divergence /= gradient_accumulation_steps

    if ProcessGroupManager.get_tensor_parallel_world_size() > 1:
        loss = loss.to_local()
        raise NotImplementedError()

    tensor = torch.stack([loss, lm_loss, kl_divergence])
    torch.distributed.all_reduce(tensor, op=ReduceOp.AVG, group=ProcessGroupManager.get_data_parallel_group())

    tensor = tensor.tolist()
    loss, lm_loss, kl_divergence = tensor
    grad_norm = 0 if grad_norm is None else grad_norm.item()

    return loss, lm_loss, kl_divergence, grad_norm


if __name__ == "__main__":
    main(Mode.distillation)
