import argparse
import os

import torch
import torch.distributed

from dolomite_engine.hf_models import AttentionHeadType, GPTDolomiteConfig, GPTDolomiteForCausalLM_TP
from dolomite_engine.hf_models.models.gpt_dolomite_TP import fix_unsharded_state_dict
from dolomite_engine.utils import CUDA_RNGStatesTracker, ProcessGroupManager, set_cuda_rng_tracker

from ...test_common import TestCommons


parser = argparse.ArgumentParser()
parser.add_argument("--attention-head-type", type=str)
parser.add_argument("--activation-function", type=str)
parser.add_argument("--tensor-parallel-word-embeddings", action="store_true")
parser.add_argument("--tmp-path", type=str)
args = parser.parse_args()


ProcessGroupManager(tensor_parallel_size=int(os.getenv("WORLD_SIZE")))

tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
cuda_rng_tracker = CUDA_RNGStatesTracker()
cuda_rng_tracker.add("tensor-parallel-seed", 42)
set_cuda_rng_tracker(cuda_rng_tracker)


num_key_value_heads = None
if AttentionHeadType(args.attention_head_type) == AttentionHeadType.gqa:
    num_key_value_heads = 8

config = GPTDolomiteConfig(
    attention_head_type=args.attention_head_type,
    n_layer=1,
    position_embedding_type="learned_absolute",
    num_key_value_heads=num_key_value_heads,
    add_bias=False,
    n_embd=128,
    n_head=16,
)

if tp_rank == 0:
    model = TestCommons.from_config(None, config)
    model.save_pretrained(args.tmp_path, safe_serialization=True)

torch.distributed.barrier()

model_tp = GPTDolomiteForCausalLM_TP.from_pretrained(
    args.tmp_path, tensor_parallel_word_embeddings=args.tensor_parallel_word_embeddings
)

tp_state_dict = model_tp.state_dict()
tp_state_dict = {key: value.to("cpu").full_tensor() for key, value in tp_state_dict.items()}
tp_state_dict = fix_unsharded_state_dict(config, tp_state_dict, ProcessGroupManager.get_tensor_parallel_world_size())

torch.distributed.barrier()

if tp_rank == 0:
    original_state_dict = model.state_dict()

    assert tp_state_dict.keys() == original_state_dict.keys()
    for key in original_state_dict:
        assert original_state_dict[key].equal(tp_state_dict[key])

ProcessGroupManager.destroy_process_groups()
