import argparse
import os

import torch
import torch.distributed

from dolomite_engine.arguments import TrainingArgs, UnshardingArgs
from dolomite_engine.checkpointing import load_checkpoint_for_inference, save_checkpoint
from dolomite_engine.distributed import wrap_model_container_for_distributed_training
from dolomite_engine.enums import Mode
from dolomite_engine.hf_models import AttentionHeadType
from dolomite_engine.model_wrapper import get_model_container
from dolomite_engine.utils import ProcessGroupManager, load_yaml


parser = argparse.ArgumentParser()
parser.add_argument("--train-config", type=str)
parser.add_argument("--unshard-config", type=str)
parser.add_argument("--attention-head-type", type=str)
parser.add_argument("--activation-function", type=str)
parser.add_argument("--tmp-path", type=str)
parser.add_argument("--tensor-parallel-word-embeddings", action="store_true")
parser.add_argument("--zero-stage", type=int)
parser.add_argument("--data-parallel-replication-world-size", type=int)
parser.add_argument("--data-parallel-sharding-world-size", type=int)
args = parser.parse_args()

num_key_value_heads = None
if AttentionHeadType(args.attention_head_type) == AttentionHeadType.gqa:
    num_key_value_heads = 8

train_config = TrainingArgs(**load_yaml(args.train_config))
unshard_config = UnshardingArgs(**load_yaml(args.unshard_config))

# set zero stage
train_config.distributed_args.stage = args.zero_stage
# set tensor parallel embeddings
train_config.distributed_args.tensor_parallel_word_embeddings = args.tensor_parallel_word_embeddings
# attention head type
train_config.model_args.pretrained_config["attention_head_type"] = args.attention_head_type
train_config.model_args.pretrained_config["num_key_value_heads"] = num_key_value_heads
# activation function
train_config.model_args.pretrained_config["activation_function"] = args.activation_function

ProcessGroupManager(
    tensor_parallel_world_size=train_config.distributed_args.tensor_parallel_world_size,
    pipeline_parallel_world_size=train_config.distributed_args.pipeline_parallel_world_size,
)

global_rank = ProcessGroupManager.get_global_rank()

if global_rank == 0:
    with (
        ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
        ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
        ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
        ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
    ):
        original_num_stages = train_config.distributed_args.num_pipeline_stages
        train_config.distributed_args.num_pipeline_stages = 1

        model_container = get_model_container(train_config, Mode.training)
        model_container[0].save_pretrained(os.path.join(args.tmp_path, "single_rank"))

        train_config.distributed_args.num_pipeline_stages = original_num_stages

torch.distributed.barrier()

# modify args to load the saved single_rank checkpoint
train_config.model_args.pretrained_config = None
train_config.model_args.model_name = os.path.join(args.tmp_path, "single_rank")
train_config.save_args.save_path = os.path.join(args.tmp_path, "save")

# modify unsharding args to load the checkpoint for unsharding
iteration = 0
unshard_config.load_args.load_path = train_config.save_args.save_path
unshard_config.load_args.iteration = iteration
unshard_config.unsharded_path = os.path.join(args.tmp_path, "unsharded_path")

parallel_model_container = get_model_container(train_config, Mode.training)
parallel_model_container, _ = wrap_model_container_for_distributed_training(train_config, parallel_model_container)

save_checkpoint(
    train_config,
    model_container=parallel_model_container,
    optimizer_container=None,
    lr_scheduler_container=None,
    train_dataloader=None,
    experiments_tracker=None,
    iteration=iteration,
    metadata=None,
)

torch.distributed.barrier()

_, _, consolidated_state_dict = load_checkpoint_for_inference(unshard_config, mode=Mode.unsharding, use_meta=False)

if global_rank == 0:
    original_state_dict = model_container[0].state_dict()

    assert consolidated_state_dict.keys() == original_state_dict.keys()
    for key in original_state_dict:
        assert original_state_dict[key].equal(consolidated_state_dict[key])

ProcessGroupManager.destroy_process_groups()
