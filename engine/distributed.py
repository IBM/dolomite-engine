from functools import partial
from typing import Tuple, Union

import torch
from accelerate import FullyShardedDataParallelPlugin as FSDP_plugin
from deepspeed import DeepSpeedEngine
from deepspeed import initialize as deepspeed_initialize
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision as FSDP_MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .arguments import TrainingArgs
from .enums import DistributedBackend
from .optimization import get_optimizer_and_lr_scheduler


_DEEPSPEED_CONFIG: dict = None


def wrap_model_for_distributed_training(
    args: TrainingArgs, model: torch.nn.Module
) -> Tuple[Union[DeepSpeedEngine, DDP, FSDP], Optimizer, LambdaLR]:
    """converts the model to a ZeRO-DP sharded model

    Args:
        args (TrainingArgs): arguments based on training mode
        model (torch.nn.Module): any torch.nn.Module object

    Returns:
        Tuple[Union[DeepSpeedEngine, DDP, FSDP], Optimizer, LambdaLR]: parallelized model, optimizer and lr_scheduler
    """

    stage = args.distributed_args.stage
    cpu_offload = args.distributed_args.cpu_offload

    if args.distributed_args.distributed_backend == DistributedBackend.deepspeed:
        assert stage in [1, 2, 3]

        optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(
            optimizer_class_name=args.optimizer_args.class_name,
            optimizer_class_args=args.optimizer_args.class_args,
            cpu_offload=cpu_offload,
            trainable_parameters=model.parameters(),
            lr_schedule=args.lr_scheduler_args.lr_schedule,
            num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
            num_training_steps=args.training_parameters.num_training_steps,
        )

        model, _, _, _ = deepspeed_initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=get_deepspeed_config(args),
        )

        # we don't need the optimizer and scheduler when using deepspeed backend
        optimizer = None
        lr_scheduler = None
    elif args.distributed_args.distributed_backend == DistributedBackend.torch:
        assert stage in [0, 2, 3]

        if stage == 0:
            assert not cpu_offload
            assert args.model_args.dtype == torch.float32

            model = DDP(model.to(torch.cuda.current_device()))
        else:
            assert not cpu_offload

            if stage == 2:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            elif stage == 3:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                raise ValueError(f"unexpected stage ({stage}) for torch backend")

            mixed_precision = FSDP_MixedPrecision(param_dtype=args.model_args.dtype)
            model = FSDP(
                model.to(torch.cuda.current_device()),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=[
                        FSDP_plugin.get_module_class_from_name(model.model, name)
                        for name in model.model._no_split_modules
                    ],
                ),
                limit_all_gathers=True,
                use_orig_params=True,
            )

        optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(
            optimizer_class_name=args.optimizer_args.class_name,
            optimizer_class_args=args.optimizer_args.class_args,
            cpu_offload=cpu_offload,
            trainable_parameters=model.parameters(),
            lr_schedule=args.lr_scheduler_args.lr_schedule,
            num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
            num_training_steps=args.training_parameters.num_training_steps,
        )

    return model, optimizer, lr_scheduler


def get_deepspeed_config(args: TrainingArgs) -> dict:
    """generate deepspeed config from the args

    Args:
        args (TrainingArgs): arguments based on training mode

    Returns:
        dict: deepspeed config
    """

    global _DEEPSPEED_CONFIG

    if _DEEPSPEED_CONFIG is None:
        config = {
            "zero_optimization": {
                "stage": args.distributed_args.stage,
                "overlap_comm": args.distributed_args.overlap_comm,
                "contiguous_gradients": args.distributed_args.contiguous_gradients,
            },
            "train_micro_batch_size_per_gpu": args.training_parameters.batch_size_per_gpu,
            "gradient_accumulation_steps": args.training_parameters.gradient_accumulation_steps,
        }

        # mixed precision options
        if args.model_args.dtype == torch.bfloat16:
            config["bf16"] = {"enabled": True}
        elif args.model_args.dtype == torch.float16:
            config["fp16"] = {"enabled": True, "auto_cast": True}

        # cpu offload
        if args.distributed_args.cpu_offload:
            config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
            config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

        _DEEPSPEED_CONFIG = config

    return _DEEPSPEED_CONFIG
