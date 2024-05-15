import argparse
import random

import torch
import torch.distributed

from dolomite_engine import (
    AttentionHeadType,
    CUDA_RNGStatesTracker,
    GPTDolomiteConfig,
    GPTDolomiteForCausalLM_TP,
    ProcessGroupManager,
    SafeTensorsWeightsManager,
    set_cuda_rng_tracker,
    set_tensor_parallel_group_manager,
)

from ..test_common import TestCommons


parser = argparse.ArgumentParser()
parser.add_argument("--attention-head-type", type=str)
parser.add_argument("--position-embedding-type", type=str)
parser.add_argument("--attention-implementation", type=str)
parser.add_argument("--tmp-path", type=str)
args = parser.parse_args()


# initialize distributed
torch.distributed.init_process_group("nccl")

device_count_per_node = torch.cuda.device_count()
local_rank = torch.distributed.get_rank() % device_count_per_node

torch.cuda.set_device(local_rank)

# this assumes all GPUs fall in tensor parallel group
tensor_parallel_manager = ProcessGroupManager()
set_tensor_parallel_group_manager(tensor_parallel_manager)

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
cuda_rng_tracker = CUDA_RNGStatesTracker()
cuda_rng_tracker.add("tensor-parallel-seed", 42)
set_cuda_rng_tracker(cuda_rng_tracker)


num_key_value_heads = None
if AttentionHeadType(args.attention_head_type) == AttentionHeadType.gqa:
    num_key_value_heads = 4

config = GPTDolomiteConfig(
    attention_head_type=args.attention_head_type,
    n_layer=1,
    position_embedding_type=args.position_embedding_type,
    num_key_value_heads=num_key_value_heads,
    add_bias=False,
    n_embd=12,
)

if torch.distributed.get_rank() == 0:
    with torch.device("meta"):
        model = TestCommons.from_config(None, config, attn_implementation=args.attention_implementation)

    model = model.to_empty(device=torch.cuda.current_device())
    for _, param in model.named_parameters():
        param.data.normal_(0, 0.0125)
    model.transformer.rope.initialize("cuda")

    model.eval()

    model.save_pretrained(args.tmp_path, safe_serialization=True)

torch.distributed.barrier()

# use dummy tensors to avoid initializing model here
with torch.device("meta"):
    # try sharding vocab matrices if really struggling for memory
    model_tp = GPTDolomiteForCausalLM_TP(
        config,
        tensor_parallel_vocab_matrix=False,
        tensor_parallel_position_embedding_matrix=False,
        attn_implementation=args.attention_implementation,
    )

    if torch.distributed.get_rank() == 0:
        print(model_tp)

# copy to device without copying storage
model_tp = model_tp.to_empty(device=torch.cuda.current_device())

# load weights into tensor parallel model using SafeTensorsWeightsManager class
# this avoids loading multiple copies of the parameters in CPU memory
safetensors_weight_manager = SafeTensorsWeightsManager(args.tmp_path)
model_tp.load_unsharded_weights(safetensors_weight_manager)
model_tp.transformer.rope.initialize("cuda")

# set model to eval mode
model_tp.eval()

random.seed(42)


with torch.inference_mode():
    x = torch.randint(0, 50255, (4, 512), device=torch.cuda.current_device(), requires_grad=False)
    y_tp = model_tp(x)

    if torch.distributed.get_rank() == 0:
        y = model(x)
        print((y[0] - y_tp[0]).abs().max())
