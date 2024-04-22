import logging
from copy import deepcopy
from functools import partial
from typing import Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..arguments import TrainingArgs
from ..enums import DistributedBackend, FP8Backend
from ..gradient_checkpointing import apply_gradient_checkpointing
from ..model_wrapper import ModelWrapper
from ..optimization import get_optimizer_and_lr_scheduler
from ..utils import get_module_class_from_name, log_rank_0, string_to_torch_dtype
from .deepspeed import get_deepspeed_config
from .fp8 import convert_model_to_transformer_engine


_STAGE_FULL_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy.SHARD_GRAD_OP,
    3: ShardingStrategy.FULL_SHARD,
}

_STAGE_HYBRID_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy._HYBRID_SHARD_ZERO2,
    3: ShardingStrategy.HYBRID_SHARD,
}

_FSDP_MIXED_PRECISION_POLICIES = {
    "fp32": MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
    "fp16": MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16),
    "bf16": MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
}


def wrap_model_for_distributed_training(
    args: TrainingArgs, model: torch.nn.Module
) -> Tuple[ModelWrapper, Optimizer, LambdaLR]:
    """converts the model to a ZeRO-DP sharded model

    Args:
        args (TrainingArgs): arguments based on training mode
        model (ModelWrapper): any torch.nn.Module object

    Returns:
        Tuple[ModelWrapper, Optimizer, LambdaLR]: parallelized model, optimizer and lr_scheduler
    """

    stage = args.distributed_args.stage
    cpu_offload = args.distributed_args.cpu_offload
    torch_compile = args.distributed_args.torch_compile
    dtype = args.mixed_precision_args.dtype
    communication_dtype = args.distributed_args.communication_dtype
    fp8_backend = args.mixed_precision_args.fp8_backend

    if dtype in ["fp16", "bf16"]:
        if communication_dtype != "fp32":
            log_rank_0(
                logging.WARN,
                f"using ({communication_dtype}) with mixed precision training in ({dtype}), recommended is to use ({torch.float32})",
            )

    if dtype == "fp8" and fp8_backend == FP8Backend.nvte:
        convert_model_to_transformer_engine(model)
        dtype = "bf16"

    assert args.distributed_args.zero_hpz_partition_size in [
        1,
        torch.cuda.device_count(),
    ], "currently we only support 1 and number of GPUs per node for HSDP"

    block_names = model.model._no_split_modules

    if args.distributed_args.distributed_backend == DistributedBackend.deepspeed:
        assert stage in [1, 2, 3]
        assert not torch_compile

        optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(
            optimizer_class_name=args.optimizer_args.class_name,
            optimizer_class_args=args.optimizer_args.class_args,
            cpu_offload=cpu_offload,
            model=model,
            num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
            num_constant_steps=args.lr_scheduler_args.num_constant_steps,
            num_decay_steps=args.lr_scheduler_args.num_decay_steps,
            num_training_steps=args.training_parameters.num_training_steps,
            lr_decay_style=args.lr_scheduler_args.lr_decay_style,
            lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
            params_group_method=args.optimizer_args.params_group_method,
        )

        if args.distributed_args.gradient_checkpointing_method is not None:
            assert len(block_names) == 1

            apply_gradient_checkpointing(
                model,
                args.distributed_args.gradient_checkpointing_method,
                block_name=block_names[0],
                **args.distributed_args.gradient_checkpointing_args,
            )

        from deepspeed import initialize as deepspeed_initialize

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
        assert not cpu_offload

        if stage == 0:
            sharding_strategy = ShardingStrategy.NO_SHARD
        else:
            if args.distributed_args.zero_hpz_partition_size == 1:
                sharding_strategy = _STAGE_FULL_SHARDING_STRATEGY_MAP[stage]
            else:
                assert args.distributed_args.zero_hpz_partition_size == 8

                sharding_strategy = _STAGE_HYBRID_SHARDING_STRATEGY_MAP[stage]

        mixed_precision_policy = deepcopy(_FSDP_MIXED_PRECISION_POLICIES[dtype])
        if communication_dtype is not None:
            mixed_precision_policy.reduce_dtype = string_to_torch_dtype(communication_dtype)

        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision_policy,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=[get_module_class_from_name(model, name) for name in block_names],
            ),
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
            sync_module_states=args.model_args.efficient_initialization,
        )

        if args.distributed_args.gradient_checkpointing_method is not None:
            assert len(block_names) == 1

            apply_gradient_checkpointing(
                model,
                args.distributed_args.gradient_checkpointing_method,
                block_name=block_names[0],
                **args.distributed_args.gradient_checkpointing_args,
            )

        if torch_compile:
            model = torch.compile(model)

        optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(
            optimizer_class_name=args.optimizer_args.class_name,
            optimizer_class_args=args.optimizer_args.class_args,
            cpu_offload=cpu_offload,
            model=model,
            num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
            num_constant_steps=args.lr_scheduler_args.num_constant_steps,
            num_decay_steps=args.lr_scheduler_args.num_decay_steps,
            num_training_steps=args.training_parameters.num_training_steps,
            lr_decay_style=args.lr_scheduler_args.lr_decay_style,
            lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
            params_group_method=args.optimizer_args.params_group_method,
        )

    return model, optimizer, lr_scheduler
