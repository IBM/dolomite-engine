import argparse
import os
import random

import torch
import torch.distributed
from transformers import AutoModelForCausalLM

from dolomite_engine.hf_models import (
    AttentionHeadType,
    GPTDolomiteConfig,
    GPTDolomiteForCausalLM_TP,
    GPTEnsembleConfig,
)
from dolomite_engine.utils import CUDA_RNGStatesTracker, ProcessGroupManager, set_cuda_rng_tracker


parser = argparse.ArgumentParser()

parser.add_argument("--attention-head-type", type=str)
parser.add_argument("--position-embedding-type", type=str)
parser.add_argument("--model-scale", type=str)
parser.add_argument("--attention-implementation", type=str)
parser.add_argument("--tensor-parallel-word-embeddings", action="store_true")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--input-sequence-tokens", type=int, default=1024)
parser.add_argument("--max-new-tokens", type=int, default=128)

args = parser.parse_args()


ProcessGroupManager(tensor_parallel_size=int(os.getenv("WORLD_SIZE")))

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
cuda_rng_tracker = CUDA_RNGStatesTracker()
cuda_rng_tracker.add("tensor-parallel-seed", 42)
set_cuda_rng_tracker(cuda_rng_tracker)


num_key_value_heads = None
if AttentionHeadType(args.attention_head_type) == AttentionHeadType.gqa:
    num_key_value_heads = 8

kwargs = dict(
    attention_head_type=args.attention_head_type,
    n_layer=88,
    position_embedding_type=args.position_embedding_type,
    num_key_value_heads=num_key_value_heads,
    add_bias=False,
    n_embd=6144,
    n_head=48,
    n_inner=24576,
    n_positions=8192,
    vocab_size=49152,
)

if args.model_scale == "34b":
    config = GPTDolomiteConfig(**kwargs)
elif args.model_scale == "34b-ensemble":
    config = GPTEnsembleConfig(**kwargs)

# use dummy tensors to avoid initializing model here
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

    num_parameters = 0
    for param_name, param in model.named_parameters():
        num_parameters += param.numel()

    print("\ntotal", f"{num_parameters:,}")

    del model

    # try sharding vocab matrices if really struggling for memory
    model_tp = GPTDolomiteForCausalLM_TP._from_config(
        config,
        tensor_parallel_word_embeddings=args.tensor_parallel_word_embeddings,
        attn_implementation=args.attention_implementation,
    )

# copy to device without copying storage
model_tp = model_tp.to_empty(device=torch.cuda.current_device())
model_tp.post_init()

# set model to eval mode
model_tp.eval()

random.seed(42)

input_ids = torch.randint(
    0, 50255, (args.batch_size, args.input_sequence_length), device=torch.cuda.current_device(), requires_grad=False
)

output_tp = model_tp.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
