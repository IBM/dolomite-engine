from copy import deepcopy

from ..arguments import TrainingArgs


_DEEPSPEED_MIXED_PRECISION_CONFIG = {
    "fp32": {},
    "fp16": {"fp16": {"enabled": True, "auto_cast": True}},
    "bf16": {"bf16": {"enabled": True}},
}


_DEEPSPEED_CONFIG: dict | None = None


def set_deepspeed_config(args: TrainingArgs) -> None:
    """sets the deepspeed config from the arguments

    Args:
        args (TrainingArgs): arguments based on training mode
    """

    config = {
        "zero_optimization": {
            "stage": args.distributed_args.stage,
            "overlap_comm": args.distributed_args.overlap_comm,
            "contiguous_gradients": args.distributed_args.contiguous_gradients,
            # whether to use quantized weights (ZeRO++)
            "zero_quantized_weights": args.distributed_args.zero_quantized_weights,
            # # whether to use quantized gradients (ZeRO++)
            "zero_quantized_gradients": args.distributed_args.zero_quantized_gradients,
        },
        "train_micro_batch_size_per_gpu": args.training_parameters.micro_batch_size,
        "gradient_accumulation_steps": args.training_parameters.gradient_accumulation_steps,
        "gradient_clipping": args.training_parameters.gradient_clipping,
    }

    if args.distributed_args.zero_topology is not None:
        config["zero_optimization"][
            "zero_hpz_partition_size"
        ] = args.distributed_args.zero_topology.data_parallel_sharding_world_size

    dtype_config: dict = deepcopy(_DEEPSPEED_MIXED_PRECISION_CONFIG[args.mixed_precision_args.dtype])
    if args.distributed_args.communication_dtype is not None:
        dtype_config.update(
            {
                "data_types": {"grad_accum_dtype": args.distributed_args.communication_dtype},
                "communication_data_type": args.distributed_args.communication_dtype,
            }
        )
    config.update(dtype_config)

    # cpu offload
    if args.distributed_args.cpu_offload:
        cpu_params = {"device": "cpu", "pin_memory": True}
        config["zero_optimization"]["offload_param"] = cpu_params
        config["zero_optimization"]["offload_optimizer"] = cpu_params

    global _DEEPSPEED_CONFIG
    _DEEPSPEED_CONFIG = config


def get_deepspeed_config() -> dict:
    """generate deepspeed config from the args

    Args:
        args (TrainingArgs): arguments based on training mode

    Returns:
        dict: deepspeed config
    """

    global _DEEPSPEED_CONFIG
    return _DEEPSPEED_CONFIG
