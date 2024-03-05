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

_STAGE_SHARDING_STRATEGY_MAP = {
    0: ShardingStrategy.NO_SHARD,
    2: ShardingStrategy.SHARD_GRAD_OP,
    3: ShardingStrategy.FULL_SHARD,
}

_FSDP_MIXED_PRECISION_POLICIES = {
    torch.float32: FSDP_MixedPrecision(
        param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
    ),
    torch.float16: FSDP_MixedPrecision(
        param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
    ),
    torch.bfloat16: FSDP_MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
    ),
}

_DEEPSPEED_MIXED_PRECISION_CONFIG = {
    torch.float32: {},
    torch.float16: {"fp16": {"enabled": True, "auto_cast": True}},
    torch.bfloat16: {"bf16": {"enabled": True}, "data_types": {"grad_accum_dtype": "fp32"}},
}


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
        assert stage in _STAGE_SHARDING_STRATEGY_MAP
        assert not cpu_offload

        model = FSDP(
            model.to(torch.cuda.current_device()),
            sharding_strategy=_STAGE_SHARDING_STRATEGY_MAP[stage],
            mixed_precision=_FSDP_MIXED_PRECISION_POLICIES[args.model_args.dtype],
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=[
                    FSDP_plugin.get_module_class_from_name(model.model, name) for name in model.model._no_split_modules
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

        config.update(_DEEPSPEED_MIXED_PRECISION_CONFIG[args.model_args.dtype])

        # cpu offload
        if args.distributed_args.cpu_offload:
            cpu_params = {"device": "cpu", "pin_memory": True}
            config["zero_optimization"]["offload_param"] = cpu_params
            config["zero_optimization"]["offload_optimizer"] = cpu_params

        _DEEPSPEED_CONFIG = config

    return _DEEPSPEED_CONFIG
