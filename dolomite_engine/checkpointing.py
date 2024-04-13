import json
import os
import random
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .arguments import ExportArgs, InferenceArgs, TrainingArgs, get_args_file_extension
from .data import DataLoader
from .enums import ArgsFileExtension, DistributedBackend, Mode, TuningMethod
from .model_wrapper import ModelWrapper, get_model
from .utils import ExperimentsTracker, get_global_rank, load_yaml, register_timer, run_rank_n


_TRAINING_CONFIG_PREFIX = "training_config"
_INFERENCE_CONFIG_PREFIX = "inference_config"


@register_timer("save_checkpoint")
def save_checkpoint(
    args: TrainingArgs,
    model: ModelWrapper,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: DataLoader,
    experiments_tracker: ExperimentsTracker,
    iteration: int,
    metadata: dict = None,
) -> None:
    """save checkpoint during training

    Args:
        args (TrainingArgs): arguments for training
        model (ModelWrapper): model to save
        optimizer (Optimizer): optimizer to save
        lr_scheduler (LambdaLR): learning rate scheduler to save
        train_dataloader (DataLoader): train dataloader to save
        experiments_tracker (ExperimentsTracker): experiment tracker to save
        iteration (int): current iteration
        metadata (dict): extra stuff to store

    Raises:
        ValueError: if unexpected distributed backend is found
    """

    distributed_backend = args.distributed_args.distributed_backend
    stage = args.distributed_args.stage

    save_path = _get_base_path(args.save_args.save_path, iteration)
    os.makedirs(save_path, exist_ok=True)

    if distributed_backend == DistributedBackend.deepspeed:
        from deepspeed import DeepSpeedEngine

        assert isinstance(model, DeepSpeedEngine)

        model.save_checkpoint(args.save_args.save_path, tag=_get_checkpoint_tag(iteration))
    elif distributed_backend == DistributedBackend.torch:
        assert isinstance(model, FSDP)

        if stage == 0:
            run_rank_n(torch.save)(model.state_dict(), _get_model_path(save_path))
            run_rank_n(torch.save)(optimizer.state_dict(), _get_optimizer_path(save_path))
        else:
            # TODO add support for local state dict
            with FSDP.state_dict_type(
                model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                run_rank_n(torch.save)(model.state_dict(), _get_model_path(save_path))
                run_rank_n(torch.save)(
                    FSDP.optim_state_dict(model=model, optim=optimizer), _get_optimizer_path(save_path)
                )

        run_rank_n(torch.save)(lr_scheduler.state_dict(), _get_lr_scheduler_path(save_path))
    else:
        raise ValueError(f"unexpected distributed_backend ({distributed_backend})")

    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    rng_state_path = _get_rng_state_path(save_path)
    os.makedirs(os.path.dirname(rng_state_path), exist_ok=True)
    torch.save(rng_state, rng_state_path)

    if train_dataloader is not None:
        dataloader_path = _get_dataloader_path(save_path)
        os.makedirs(os.path.dirname(dataloader_path), exist_ok=True)
        torch.save(train_dataloader.state_dict(), dataloader_path)

    if experiments_tracker is not None:
        run_rank_n(json.dump)(
            experiments_tracker.state_dict(), run_rank_n(open)(_get_experiments_tracker_path(save_path), "w"), indent=4
        )

    if metadata is not None:
        run_rank_n(json.dump)(metadata, run_rank_n(open)(_get_metadata_path(save_path), "w"), indent=4)

    save_args(args, save_path, mode=Mode.training)

    dist.barrier()

    run_rank_n(json.dump)(
        {"latest_checkpointed_iteration": iteration},
        run_rank_n(open)(_get_latest_checkpointed_iterations_path(args.save_args.save_path), "w"),
        indent=4,
    )


@register_timer("load_checkpoint_for_training")
def load_checkpoint_for_training(
    args: TrainingArgs,
    model: ModelWrapper,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: DataLoader,
) -> Tuple[int, dict]:
    """load checkpoint for training

    Args:
        args (TrainingArgs): arguments for training
        model (ModelWrapper): model to load
        optimizer (Optimizer): optimizer to save
        lr_scheduler (LambdaLR): learning rate scheduler to load
        train_dataloader (DataLoader): train dataloader to load

    Raises:
        ValueError: if unexpected distributed backend is found

    Returns:
        Tuple[int, dict, dict]: checkpointed iteration, metadata, experiments_tracker state dict
    """

    if args.load_args is None or args.load_args.load_path is None:
        return

    distributed_backend = args.distributed_args.distributed_backend
    stage = args.distributed_args.stage

    iteration = args.load_args.iteration
    if iteration is None:
        iteration = json.load(open(_get_latest_checkpointed_iterations_path(args.load_args.load_path), "r"))[
            "latest_checkpointed_iteration"
        ]

    load_path = _get_base_path(args.load_args.load_path, iteration)

    if distributed_backend == DistributedBackend.deepspeed:
        from deepspeed import DeepSpeedEngine

        assert isinstance(model, DeepSpeedEngine)

        model.load_checkpoint(args.load_args.load_path, tag=_get_checkpoint_tag(iteration))
    elif distributed_backend == DistributedBackend.torch:
        assert isinstance(model, FSDP)

        if stage == 0:
            model.load_state_dict(torch.load(_get_model_path(load_path)))
            optimizer.load_state_dict(torch.load(_get_optimizer_path(load_path)))
        else:
            # TODO add support for local state dict
            with FSDP.state_dict_type(
                model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                model.load_state_dict(torch.load(_get_model_path(load_path)))
                optimizer.load_state_dict(
                    FSDP.optim_state_dict_to_load(
                        model=model, optim=optimizer, optim_state_dict=torch.load(_get_optimizer_path(load_path))
                    )
                )

        lr_scheduler.load_state_dict(torch.load(_get_lr_scheduler_path(load_path)))
    else:
        raise ValueError(f"unexpected distributed_backend ({distributed_backend})")

    rng_state = torch.load(_get_rng_state_path(load_path))
    random.setstate(rng_state["random_rng_state"])
    np.random.set_state(rng_state["np_rng_state"])
    torch.set_rng_state(rng_state["torch_rng_state"])
    torch.cuda.set_rng_state(rng_state["cuda_rng_state"])

    metadata = None
    if os.path.isfile(_get_metadata_path(load_path)):
        metadata = json.load(open(_get_metadata_path(load_path), "r"))

    if train_dataloader is not None:
        train_dataloader.load_state_dict(torch.load(_get_dataloader_path(load_path)))

    experiments_tracker_json = None
    if os.path.exists(_get_experiments_tracker_path(load_path)):
        experiments_tracker_json = json.load(open(_get_experiments_tracker_path(load_path), "r"))

    return iteration, metadata, experiments_tracker_json


def load_checkpoint_for_inference(
    args: Union[InferenceArgs, ExportArgs], mode: Mode
) -> Tuple[ModelWrapper, TrainingArgs]:
    """load deepspeed checkpoint for inference

    Args:
        args (Union[InferenceArgs, ExportArgs]): arguments
        mode (Mode): training/inference mode
    """

    load_path = args.load_args.load_path
    iteration = args.load_args.iteration

    args_file = os.path.join(_get_base_path(load_path, iteration), f"{_TRAINING_CONFIG_PREFIX}.json")
    if os.path.isfile(args_file):
        args_from_checkpoint = json.load(open(args_file, "r"))
    else:
        args_file = os.path.join(_get_base_path(load_path, iteration), f"{_TRAINING_CONFIG_PREFIX}.yaml")
        args_from_checkpoint = load_yaml(args_file)

    args_from_checkpoint: TrainingArgs = TrainingArgs(**args_from_checkpoint)

    distributed_backend = args_from_checkpoint.distributed_args.distributed_backend

    model = get_model(args_from_checkpoint, mode)
    model = model.to(model.input_device)

    if distributed_backend == DistributedBackend.deepspeed:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        state = get_fp32_state_dict_from_zero_checkpoint(load_path, _get_checkpoint_tag(iteration))

        if model.tuning_method == TuningMethod.prompt_tuning:
            model.load_state_dict(state, strict=False)
        elif model.tuning_method in [TuningMethod.pretraining, TuningMethod.full_finetuning]:
            for key in state:
                state[key] = state[key].to(model.dtype)

            model.load_state_dict(state)
    elif distributed_backend == DistributedBackend.torch:
        model.load_state_dict(torch.load(_get_model_path(_get_base_path(load_path, iteration))))
    else:
        raise ValueError(f"unexpected distributed_backend ({args['distributed_args']['distributed_backend']})")

    return model, args_from_checkpoint


@run_rank_n
def save_args(args: Union[TrainingArgs, InferenceArgs], save_path: str, mode: Mode) -> None:
    """saves training args as a json

    Args:
        args (Union[TrainingArgs, InferenceArgs]): arguments for training or inference
        save_path (str): save location on disk
    """

    args_file_extension = get_args_file_extension()
    args = args.to_dict()

    file_prefix = _TRAINING_CONFIG_PREFIX if mode == Mode.training else _INFERENCE_CONFIG_PREFIX
    save_path = os.path.join(save_path, f"{file_prefix}.{args_file_extension.value}")

    if args_file_extension == ArgsFileExtension.json:
        json.dump(args, open(save_path, "w"), indent=4)
    elif args_file_extension == ArgsFileExtension.yaml:
        yaml.dump(args, open(save_path, "w"), indent=2)
    else:
        raise ValueError(f"unexpected file extension ({args_file_extension})")


def _get_checkpoint_tag(iteration: int) -> str:
    return f"global_step{iteration}"


def _get_base_path(path: str, iteration: int) -> str:
    return os.path.join(path, _get_checkpoint_tag(iteration))


def _get_model_path(path: str) -> str:
    return os.path.join(path, "model.pt")


def _get_optimizer_path(path: str) -> str:
    return os.path.join(path, "optimizer.pt")


def _get_lr_scheduler_path(path: str) -> str:
    return os.path.join(path, "lr_scheduler.pt")


def _get_dataloader_path(path: str) -> str:
    return os.path.join(path, "dataloader", f"dataloader-{get_global_rank()}.pt")


def _get_rng_state_path(path: str) -> str:
    return os.path.join(path, "rng_state", f"rng_state-{get_global_rank()}.pt")


def _get_latest_checkpointed_iterations_path(path: str) -> str:
    return os.path.join(path, "latest_checkpointed_iteration.json")


def _get_experiments_tracker_path(path: str) -> str:
    return os.path.join(path, "experiments_tracker.json")


def _get_metadata_path(path: str) -> str:
    return os.path.join(path, "metadata.json")
