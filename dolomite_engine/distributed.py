from copy import deepcopy
from functools import partial
from typing import Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision as FSDP_MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .arguments import TrainingArgs
from .enums import DistributedBackend
from .gradient_checkpointing import apply_gradient_checkpointing
from .model_wrapper import ModelWrapper
from .optimization import get_optimizer_and_lr_scheduler
from .utils import get_module_class_from_name, warn_rank_0


_DEEPSPEED_CONFIG: dict = None

_STAGE_FULL_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy.SHARD_GRAD_OP,
    3: ShardingStrategy.FULL_SHARD,
}

_STAGE_HYBRID_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy._HYBRID_SHARD_ZERO2,
    3: ShardingStrategy.HYBRID_SHARD,
}

_FSDP_MIXED_PRECISION_POLICIES = {
    torch.float32: FSDP_MixedPrecision(
        param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
    ),
    torch.float16: FSDP_MixedPrecision(
        param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
    ),
    torch.bfloat16: FSDP_MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
    ),
}

_DEEPSPEED_MIXED_PRECISION_CONFIG = {
    torch.float32: {},
    torch.float16: {"fp16": {"enabled": True, "auto_cast": True}},
    torch.bfloat16: {"bf16": {"enabled": True}},
}

_TORCH_DTYPE_TO_STR = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
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

    if args.model_args.dtype in [torch.float16, torch.bfloat16]:
        if args.distributed_args.communication_dtype != torch.float32:
            warn_rank_0(
                f"using ({args.distributed_args.communication_dtype}) with mixed precision training in "
                f"({args.model_args.dtype}), recommended is to use ({torch.float32})"
            )

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
            trainable_parameters=model.parameters(),
            num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
            num_constant_steps=args.lr_scheduler_args.num_constant_steps,
            num_decay_steps=args.lr_scheduler_args.num_decay_steps,
            num_training_steps=args.training_parameters.num_training_steps,
            lr_decay_style=args.lr_scheduler_args.lr_decay_style,
            lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
        )

        from deepspeed import initialize as deepspeed_initialize

        model, _, _, _ = deepspeed_initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=get_deepspeed_config(args),
        )

        if args.distributed_args.gradient_checkpointing_method is not None:
            assert len(block_names) == 1

            apply_gradient_checkpointing(
                model,
                args.distributed_args.gradient_checkpointing_method,
                block_name=block_names[0],
                **args.distributed_args.gradient_checkpointing_args,
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

        mixed_precision_policy = deepcopy(_FSDP_MIXED_PRECISION_POLICIES[args.model_args.dtype])
        if args.distributed_args.communication_dtype is not None:
            mixed_precision_policy.reduce_dtype = args.distributed_args.communication_dtype

        efficient_cpu_initialization = args.model_args.efficient_cpu_initialization

        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=_FSDP_MIXED_PRECISION_POLICIES[args.model_args.dtype],
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=[get_module_class_from_name(model, name) for name in block_names],
            ),
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
            sync_module_states=efficient_cpu_initialization,
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
            trainable_parameters=model.parameters(),
            num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
            num_constant_steps=args.lr_scheduler_args.num_constant_steps,
            num_decay_steps=args.lr_scheduler_args.num_decay_steps,
            num_training_steps=args.training_parameters.num_training_steps,
            lr_decay_style=args.lr_scheduler_args.lr_decay_style,
            lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
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
                # hierarchical partioning for ZeRO (HSDP)
                "zero_hpz_partition_size": args.distributed_args.zero_hpz_partition_size,
                # whether to use quantized weights (ZeRO++)
                "zero_quantized_weights": args.distributed_args.zero_quantized_weights,
                # # whether to use quantized gradients (ZeRO++)
                "zero_quantized_gradients": args.distributed_args.zero_quantized_gradients,
            },
            "train_micro_batch_size_per_gpu": args.training_parameters.batch_size_per_gpu,
            "gradient_accumulation_steps": args.training_parameters.gradient_accumulation_steps,
            "gradient_clipping": args.training_parameters.gradient_clipping,
        }

        dtype_config: dict = deepcopy(_DEEPSPEED_MIXED_PRECISION_CONFIG[args.model_args.dtype])
        if args.distributed_args.communication_dtype is not None:
            dtype_str = _TORCH_DTYPE_TO_STR[args.distributed_args.communication_dtype]
            dtype_config.update({"data_types": {"grad_accum_dtype": dtype_str}, "communication_data_type": dtype_str})
        config.update(dtype_config)

        # cpu offload
        if args.distributed_args.cpu_offload:
            cpu_params = {"device": "cpu", "pin_memory": True}
            config["zero_optimization"]["offload_param"] = cpu_params
            config["zero_optimization"]["offload_optimizer"] = cpu_params

        _DEEPSPEED_CONFIG = config

    return _DEEPSPEED_CONFIG
