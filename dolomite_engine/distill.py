from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training
from .data import get_megatron_gpt_dataloaders
from .distributed import set_deepspeed_config, wrap_model_for_distributed_training
from .enums import DistributedBackend, Mode
from .model_wrapper import get_model, log_model
from .optimization import get_optimizer, get_scheduler
from .pretrain import main, train
from .utils import ExperimentsTracker, ProcessGroupManager, init_distributed, setup_tf32


def main() -> None:
    """main program"""

    mode = Mode.distillation

    setup_tf32()

    args: TrainingArgs = get_args(mode)

    # initialize distributed with nccl for multi-node communications
    init_distributed(
        tensor_parallel_size=args.distributed_args.tensor_parallel_size,
        data_parallel_size=args.distributed_args.data_parallel_size,
        data_parallel_replication_world_size=args.distributed_args.zero_topology.data_parallel_replication_world_size,
        data_parallel_sharding_world_size=args.distributed_args.zero_topology.data_parallel_sharding_world_size,
        zero_stage=args.distributed_args.stage,
        timeout_minutes=args.distributed_args.timeout_minutes,
    )
    set_seed(args.random_args.seed)

    if args.distributed_args.distributed_backend == DistributedBackend.deepspeed:
        set_deepspeed_config(args)

    model = get_model(args, mode)
    model = wrap_model_for_distributed_training(args, model)

    if args.distributed_args.distributed_backend == DistributedBackend.torch:
        optimizer = get_optimizer(
            optimizer_class_name=args.optimizer_args.class_name,
            optimizer_class_args=args.optimizer_args.class_args,
            model=model,
            params_group_method=args.optimizer_args.params_group_method,
            is_distillation=True,
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
    else:
        optimizer = None
        lr_scheduler = None

    log_model(model)

    starting_iteration = 0
    metadata = None
    experiments_tracker_state_dict = None
    if args.load_args is not None:
        starting_iteration, metadata, experiments_tracker_state_dict = load_checkpoint_for_training(
            args, model, optimizer, lr_scheduler, None
        )

        # metadata field contains the dataloader state so we need to reset it here
        if not args.load_args.load_dataloader_state and metadata is not None:
            metadata["consumed_samples"] = 0

    train_dataloader, val_dataloaders, test_dataloaders = get_megatron_gpt_dataloaders(
        args, model.tokenizer, 0 if metadata is None else metadata["consumed_samples"]
    )

    experiments_tracker = ExperimentsTracker(
        args.logging_args.experiments_tracker_name,
        args.logging_args.aim_args,
        args.logging_args.wandb_args,
        checkpoint_metadata=experiments_tracker_state_dict,
    )
    # track all hyperparams in args
    experiments_tracker.log_args(args)

    # main training loop
    train(
        args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        test_dataloaders=test_dataloaders,
        experiments_tracker=experiments_tracker,
        starting_iteration=starting_iteration,
    )

    ProcessGroupManager.destroy_process_groups()


if __name__ == "__main__":
    main()
