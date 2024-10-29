import logging
from functools import partial
from typing import Callable

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
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    _PipelineSchedule,
    get_schedule_class,
)

from ..arguments import TrainingArgs
from ..containers import ModelContainer
from ..enums import FP8Backend
from ..gradient_checkpointing import apply_gradient_checkpointing
from ..utils import ProcessGroupManager, get_module_class_from_name, log_rank_0, string_to_torch_dtype
from .fp8 import convert_model_to_transformer_engine


_STAGE_FULL_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy.SHARD_GRAD_OP,
    3: ShardingStrategy.FULL_SHARD,
}

_STAGE_HYBRID_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy._HYBRID_SHARD_ZERO2,
    3: ShardingStrategy.HYBRID_SHARD,
}


def wrap_model_container_for_distributed_training(
    args: TrainingArgs, model_container: ModelContainer
) -> tuple[ModelContainer, _PipelineSchedule]:
    """converts the model to a ZeRO-DP sharded model

    Args:
        args (TrainingArgs): arguments based on training mode
        model_container (ModelContainer): model container

    Returns:
        tuple[ModelContainer, _PipelineSchedule]: container of parallelized models and pipeline schedule
    """

    stage = args.distributed_args.stage
    cpu_offload = args.distributed_args.cpu_offload
    torch_compile = args.distributed_args.torch_compile
    dtype = args.mixed_precision_args.dtype
    communication_dtype = args.distributed_args.communication_dtype
    fp8_backend = args.mixed_precision_args.fp8_backend
    efficient_initialization = args.model_args.efficient_initialization
    fsdp_algorithm = args.distributed_args.fsdp_algorithm
    num_pipeline_stages = args.distributed_args.num_pipeline_stages

    if dtype in ["fp16", "bf16"]:
        if communication_dtype != "fp32":
            log_rank_0(
                logging.WARN,
                f"using ({communication_dtype}) with mixed precision training in ({dtype}), recommended is to use ({torch.float32})",
            )

    if dtype == "fp8" and fp8_backend == FP8Backend.nvte:
        # FIXME this wont work
        convert_model_to_transformer_engine(model)
        dtype = "bf16"

    block_names = model_container[0].model._no_split_modules
    teacher_block_names = (
        model_container[0].teacher_model._no_split_modules if model_container[0].has_teacher_model() else []
    )

    dtype = None if dtype is None else string_to_torch_dtype(dtype)
    communication_dtype = None if communication_dtype is None else string_to_torch_dtype(communication_dtype)

    assert stage in [0, 2, 3]

    dp_mesh = ProcessGroupManager.get_data_parallel_mesh()
    block_classes = [
        get_module_class_from_name(model_container[0], name) for name in block_names + teacher_block_names
    ]

    if args.distributed_args.gradient_checkpointing_method is not None:
        assert len(block_names) == 1

        for model in model_container:
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

        for i, model in enumerate(model_container):
            model_container[i] = FSDP(
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

            for i, model in enumerate(model_container):
                model_container[i] = FSDP(
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

            for i, model in enumerate(model_container):
                for module in model.modules():
                    if isinstance(module, tuple(block_classes)):
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

        for i in range(len(model_container)):
            model_container[i] = torch.compile(model_container[i])

    pipeline_stages = []
    pipeline_schedule = None

    if num_pipeline_stages > 1:
        micro_batch_size = args.training_parameters.micro_batch_size
        sequence_length = args.datasets[0].class_args.get("sequence_length")

        for model in model_container:
            intermediate_dtype = string_to_torch_dtype(args.mixed_precision_args.dtype)

            dummy_input_tensor = model.model.get_dummy_input_tensor(
                micro_batch_size, sequence_length, intermediate_dtype=intermediate_dtype
            )
            dummy_output_tensor = model.model.get_dummy_output_tensor(
                micro_batch_size,
                sequence_length,
                intermediate_dtype=intermediate_dtype,
                output_parallel_lm_logits_if_possible=True,
            )

            stage = PipelineStage(
                model,
                stage_index=model.pipeline_stage_id,
                num_stages=num_pipeline_stages,
                device=torch.cuda.current_device(),
                input_args=dummy_input_tensor,
                output_args=dummy_output_tensor,
                group=ProcessGroupManager.get_pipeline_parallel_group(),
            )
            pipeline_stages.append(stage)

        pipeline_schedule = _get_pipeline_parallel_schedule(
            pipeline_parallel_schedule=args.distributed_args.pipeline_parallel_schedule,
            gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
            pipeline_stages=pipeline_stages,
            loss_fn=model.get_loss,
        )

    return model_container, pipeline_schedule


def _get_pipeline_parallel_schedule(
    pipeline_parallel_schedule: str,
    gradient_accumulation_steps: int,
    pipeline_stages: list[PipelineStage],
    loss_fn: Callable,
) -> _PipelineSchedule:
    try:
        schedule_class = get_schedule_class(pipeline_parallel_schedule)
    except ValueError:
        raise ValueError(
            f"unexpected schedule ({pipeline_parallel_schedule}), expected values are: ['1F1B', "
            "'Interleaved1F1B', 'GPipe', 'FlexibleInterleaved1F1B', 'LoopedBFS', 'InterleavedZeroBubble', "
            "'PipelineScheduleSingle', 'PipelineScheduleMulti']"
        )

    if schedule_class in [PipelineScheduleSingle, PipelineScheduleMulti]:
        raise NotImplementedError()

    if issubclass(schedule_class, PipelineScheduleSingle):
        assert len(pipeline_stages) == 1

    def custom_loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_dict = loss_fn(output, target)
        return loss_dict["loss"]

    return schedule_class(
        pipeline_stages if issubclass(schedule_class, PipelineScheduleMulti) else pipeline_stages[0],
        n_microbatches=gradient_accumulation_steps,
        loss_fn=custom_loss_function,
    )


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
