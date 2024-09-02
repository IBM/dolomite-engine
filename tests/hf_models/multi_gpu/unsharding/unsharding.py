import argparse
import os

import torch
import torch.distributed
from torch.distributed._tensor.api import DTensor

from dolomite_engine.hf_models import (
    AttentionHeadType,
    GPTDolomiteConfig,
    GPTDolomiteForCausalLM_TP,
    GPTEnsembleConfig,
    GPTEnsembleForCausalLM_TP,
    fix_unsharded_state_dict,
    unshard_tensor_parallel_state_dicts,
)
from dolomite_engine.utils import ProcessGroupManager

from ...test_common import TestCommons


parser = argparse.ArgumentParser()
parser.add_argument("--attention-head-type", type=str)
parser.add_argument("--activation-function", type=str)
parser.add_argument("--tensor-parallel-word-embeddings", action="store_true")
parser.add_argument("--tmp-path", type=str)
parser.add_argument("--model-type", type=str)
args = parser.parse_args()


ProcessGroupManager(tensor_parallel_size=int(os.getenv("WORLD_SIZE")))

tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

num_key_value_heads = None
if AttentionHeadType(args.attention_head_type) == AttentionHeadType.gqa:
    num_key_value_heads = 8

if args.model_type == "gpt_dolomite":
    config = GPTDolomiteConfig(
        attention_head_type=args.attention_head_type,
        n_layer=1,
        position_embedding_type="learned_absolute",
        num_key_value_heads=num_key_value_heads,
        add_bias=False,
        n_embd=128,
        n_head=16,
    )
elif args.model_type == "gpt_ensemble":
    config = GPTEnsembleConfig(
        attention_head_type=args.attention_head_type,
        n_layer=1,
        position_embedding_type="learned_absolute",
        num_key_value_heads=num_key_value_heads,
        normalization_function="rmsnorm",
        resid_pdrop=0,
        add_bias=False,
        n_embd=128,
        n_head=16,
        pretraining_tensor_parallel_size=8,
    )

if tp_rank == 0:
    model = TestCommons.from_config(None, config)
    model.save_pretrained(args.tmp_path, safe_serialization=True)

torch.distributed.barrier()

if args.model_type == "gpt_dolomite":
    model_tp = GPTDolomiteForCausalLM_TP.from_pretrained(
        args.tmp_path, tensor_parallel_word_embeddings=args.tensor_parallel_word_embeddings
    )
else:
    model_tp = GPTEnsembleForCausalLM_TP.from_pretrained(
        args.tmp_path, tensor_parallel_word_embeddings=args.tensor_parallel_word_embeddings
    )

tp_state_dict = model_tp.state_dict()


def run_check(fix: bool):
    cpu_state_dict = {key: value.to("cpu") for key, value in tp_state_dict.items()}

    if fix:
        tp_state_dict_unsharded = {
            key: value.full_tensor() if isinstance(value, DTensor) else value for key, value in cpu_state_dict.items()
        }
        tp_state_dict_unsharded = fix_unsharded_state_dict(
            config, tp_state_dict_unsharded, ProcessGroupManager.get_tensor_parallel_world_size()
        )
    else:
        cpu_state_dict = {
            key: value.to_local() if isinstance(value, DTensor) else value for key, value in cpu_state_dict.items()
        }
        torch.save(
            cpu_state_dict, os.path.join(args.tmp_path, f"tp-{ProcessGroupManager.get_tensor_parallel_rank()}.pt")
        )
        del cpu_state_dict

        torch.distributed.barrier()

        tensor_parallel_state_dicts = [
            torch.load(os.path.join(args.tmp_path, f"tp-{i}.pt"))
            for i in range(ProcessGroupManager.get_tensor_parallel_world_size())
        ]

        tp_state_dict_unsharded = unshard_tensor_parallel_state_dicts(
            config,
            tensor_parallel_state_dicts=tensor_parallel_state_dicts,
            tensor_parallel_word_embeddings=args.tensor_parallel_word_embeddings,
        )

    torch.distributed.barrier()

    if tp_rank == 0:
        original_state_dict = model.state_dict()

        assert tp_state_dict_unsharded.keys() == original_state_dict.keys()
        for key in original_state_dict:
            assert original_state_dict[key].equal(tp_state_dict_unsharded[key])


if args.model_type == "gpt_dolomite":
    run_check(True)

run_check(False)

ProcessGroupManager.destroy_process_groups()
