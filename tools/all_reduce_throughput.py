import argparse
import os
import random
from time import perf_counter

import torch
import torch.distributed

from dolomite_engine.utils import ProcessGroupManager


parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", type=int)
parser.add_argument("--sequence-length", type=int)
parser.add_argument("--hidden-size", type=int)

args = parser.parse_args()

ProcessGroupManager(tensor_parallel_size=int(os.getenv("WORLD_SIZE")))

random.seed(42)

x = torch.randn(
    args.batch_size, args.sequence_length, args.hidden_size, dtype=torch.bfloat16, device=torch.cuda.current_device()
)

n = 10

with torch.inference_mode():
    for _ in range(10):
        torch.distributed.all_reduce(x, group=ProcessGroupManager.get_tensor_parallel_group())

    torch.cuda.synchronize()
    start_time = perf_counter()

    for _ in range(n):
        torch.distributed.all_reduce(x, group=ProcessGroupManager.get_tensor_parallel_group())

    torch.cuda.synchronize()
    end_time = perf_counter()

    if ProcessGroupManager.get_global_rank() == 0:
        print(f"time taken = {(end_time - start_time) / n} sec")
