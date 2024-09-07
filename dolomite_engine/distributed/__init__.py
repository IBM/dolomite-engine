import logging
from functools import partial

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy
from torch.distributed._composable.fsdp import MixedPrecisionPolicy as MixedPrecision2
from torch.distributed._composable.fsdp import OffloadPolicy, fully_shard
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision as MixedPrecision1
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from ..arguments import TrainingArgs
from ..enums import DistributedBackend, FP8Backend
from ..gradient_checkpointing import apply_gradient_checkpointing
from ..model_wrapper import ModelWrapper
from ..optimization import get_optimizer, get_scheduler
from ..utils import ProcessGroupManager, get_module_class_from_name, log_rank_0, string_to_torch_dtype
from .deepspeed import get_deepspeed_config, set_deepspeed_config
from .fp8 import convert_model_to_transformer_engine


_STAGE_FULL_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy.SHARD_GRAD_OP,
    3: ShardingStrategy.FULL_SHARD,
}

_STAGE_HYBRID_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy._HYBRID_SHARD_ZERO2,
    3: ShardingStrategy.HYBRID_SHARD,
}


def wrap_model_for_distributed_training(args: TrainingArgs, model: nn.Module) -> ModelWrapper:
    """converts the model to a ZeRO-DP sharded model

    Args:
        args (TrainingArgs): arguments based on training mode
        model (ModelWrapper): any nn.Module object

    Returns:
        ModelWrapper: parallelized model
    """

    stage = args.distributed_args.stage
    cpu_offload = args.distributed_args.cpu_offload
    torch_compile = args.distributed_args.torch_compile
    dtype = args.mixed_precision_args.dtype
    communication_dtype = args.distributed_args.communication_dtype
    fp8_backend = args.mixed_precision_args.fp8_backend
    efficient_initialization = args.model_args.efficient_initialization
    fsdp_algorithm = args.distributed_args.fsdp_algorithm

    if dtype in ["fp16", "bf16"]:
        if communication_dtype != "fp32":
            log_rank_0(
                logging.WARN,
                f"using ({communication_dtype}) with mixed precision training in ({dtype}), recommended is to use ({torch.float32})",
            )

    if dtype == "fp8" and fp8_backend == FP8Backend.nvte:
        convert_model_to_transformer_engine(model)
        dtype = "bf16"

    block_names = model.model._no_split_modules
    teacher_block_names = model.teacher_model._no_split_modules if hasattr(model, "teacher_model") else []

    dtype = None if dtype is None else string_to_torch_dtype(dtype)
    communication_dtype = None if communication_dtype is None else string_to_torch_dtype(communication_dtype)

    if args.distributed_args.distributed_backend == DistributedBackend.deepspeed:
        log_rank_0(logging.INFO, "using DeepSpeed")

        assert stage in [1, 2, 3]
        assert not torch_compile
        assert ProcessGroupManager.get_tensor_parallel_world_size() == 1

        optimizer = get_optimizer(
            optimizer_class_name=args.optimizer_args.class_name,
            optimizer_class_args=args.optimizer_args.class_args,
            model=model,
            params_group_method=args.optimizer_args.params_group_method,
        )

        lr_scheduler = get_scheduler(
            optimizer=optimizer,
            num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
            num_constant_steps=args.lr_scheduler_args.num_constant_steps,
            num_decay_steps=args.lr_scheduler_args.num_decay_steps,
            num_training_steps=args.training_parameters.num_training_steps,
            lr_decay_style=args.lr_scheduler_args.lr_decay_style,
            lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
            extra_lr_scheduler_args=args.lr_scheduler_args.extra_lr_scheduler_args,
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
            config=get_deepspeed_config(),
        )

        if args.distributed_args.gradient_checkpointing_method is not None:
            assert len(block_names) == 1

            apply_gradient_checkpointing(
                model,
                args.distributed_args.gradient_checkpointing_method,
                block_name=block_names[0],
                **args.distributed_args.gradient_checkpointing_args,
            )
    elif args.distributed_args.distributed_backend == DistributedBackend.torch:
        assert stage in [0, 2, 3]

        dp_mesh = ProcessGroupManager.get_data_parallel_mesh()
        block_classes = [get_module_class_from_name(model, name) for name in block_names + teacher_block_names]

        if args.distributed_args.gradient_checkpointing_method is not None:
            assert len(block_names) == 1

            apply_gradient_checkpointing(
                model,
                args.distributed_args.gradient_checkpointing_method,
                block_name=block_names[0],
                **args.distributed_args.gradient_checkpointing_args,
            )

        if fsdp_algorithm == 1:
            if stage == 0:
                log_rank_0(logging.INFO, "using DDP")

                assert not efficient_initialization

                sharding_strategy = ShardingStrategy.NO_SHARD
            else:
                log_rank_0(logging.INFO, "using FSDP-1")

                sharding_strategy = (
                    _STAGE_HYBRID_SHARDING_STRATEGY_MAP[stage]
                    if args.distributed_args.zero_topology.data_parallel_sharding_world_size == 8
                    else _STAGE_FULL_SHARDING_STRATEGY_MAP[stage]
                )

            def _param_init(module: nn.Module) -> None:
                assert len(teacher_block_names) == 0, "efficient initialization doesn't support distillation"

                if args.model_args.model_name is None:
                    module = module.to_empty(device=torch.cuda.current_device())

                    if hasattr(module, "reset_parameters"):
                        with torch.no_grad():
                            module.reset_parameters()
                else:
                    if efficient_initialization and ProcessGroupManager.get_data_parallel_rank() != 0:
                        module = module.to_empty(device=torch.cuda.current_device())

            model = FSDP(
                model,
                sharding_strategy=sharding_strategy,
                cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
                mixed_precision=_get_fsdp_mixed_precision(
                    dtype=dtype,
                    communication_dtype=communication_dtype,
                    fsdp_algorithm=1,
                ),
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=block_classes),
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
                use_orig_params=True,
                # https://github.com/meta-llama/llama-recipes/blob/492455dc080f6c25f356e283e443be0cce86aaeb/src/llama_recipes/finetuning.py#L191
                sync_module_states=efficient_initialization,
                param_init_fn=_param_init if efficient_initialization else None,
                device_mesh=dp_mesh,
            )
        else:
            if stage == 0:
                log_rank_0(logging.INFO, "using DDP")

                assert not efficient_initialization

                model = FSDP(
                    model,
                    sharding_strategy=ShardingStrategy.NO_SHARD,
                    cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
                    mixed_precision=_get_fsdp_mixed_precision(
                        dtype=dtype,
                        communication_dtype=communication_dtype,
                        fsdp_algorithm=1,
                    ),
                    device_id=torch.cuda.current_device(),
                    limit_all_gathers=True,
                    use_orig_params=True,
                    device_mesh=dp_mesh,
                )
            else:
                log_rank_0(logging.INFO, "using FSDP-2")

                mixed_precision_policy = _get_fsdp_mixed_precision(
                    dtype=dtype,
                    communication_dtype=communication_dtype,
                    fsdp_algorithm=2,
                )

                zero3 = stage == 3

                for module in model.modules():
                    if isinstance(module, block_classes):
                        fully_shard(
                            module,
                            mesh=dp_mesh,
                            reshard_after_forward=zero3,
                            mp_policy=mixed_precision_policy,
                            offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy(),
                        )

                fully_shard(
                    model,
                    mesh=dp_mesh,
                    reshard_after_forward=zero3,
                    mp_policy=mixed_precision_policy,
                    offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy(),
                )

                if efficient_initialization and args.model_args.model_name is None:
                    model = model.to_empty(device=torch.cuda.current_device())

                    for module in model.modules():
                        if hasattr(module, "reset_parameters"):
                            module.reset_parameters()

        if torch_compile:
            log_rank_0(logging.INFO, "using torch compile")
            model = torch.compile(model)

    return model


def _get_fsdp_mixed_precision(
    dtype: torch.dtype, communication_dtype: torch.dtype | None, fsdp_algorithm: int
) -> MixedPrecision1:
    if communication_dtype is None:
        communication_dtype = dtype

    if fsdp_algorithm == 1:
        mixed_precision = MixedPrecision1(param_dtype=dtype, reduce_dtype=communication_dtype, buffer_dtype=dtype)
    else:
        mixed_precision = MixedPrecision2(param_dtype=dtype, reduce_dtype=communication_dtype)

    return mixed_precision
