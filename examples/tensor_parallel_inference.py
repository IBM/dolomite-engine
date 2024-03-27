import torch
import torch.distributed
from transformers import AutoTokenizer

from dolomite_engine import (
    CUDA_RNGStatesTracker,
    GPTMegatronForCausalLM_TP,
    ProcessGroupManager,
    set_cuda_rng_tracker,
    set_tensor_parallel_group_manager,
)


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


model_name = "checkpoints-13b-8k/iter_00300000"

model = GPTMegatronForCausalLM_TP.from_pretrained(model_name)

# set model to eval mode
model.eval()

# happy generation
text = "def generate():"
tokenizer = AutoTokenizer.from_pretrained(model_name)

x = tokenizer([text], return_tensors="pt")

for i in x:
    x[i] = x[i].to(torch.cuda.current_device())

y = model.generate(**x, max_new_tokens=100)

if torch.distributed.get_rank() == 0:
    print(tokenizer.batch_decode(y))
