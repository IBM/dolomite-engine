import os

import scattermoe
import torch
import torch.distributed
from torch import nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from transformers import set_seed

from dolomite_engine.hf_models.modeling_utils_TP.TP import (
    dtensor_to_tensor,
    get_module_placements,
    modify_state_dict_to_dtensor_dict,
    tensor_parallel_split_safetensor_slice,
    tensor_to_dtensor,
)
from dolomite_engine.hf_models.models.moe_dolomite.config import MoEDolomiteConfig
from dolomite_engine.hf_models.models.moe_dolomite.moe.scatter import ParameterizedScatteredExperts, ScatterMoE
from dolomite_engine.hf_models.models.moe_dolomite_TP.moe_TP.scatter import ScatterMoE_TP
from dolomite_engine.utils import ProcessGroupManager


def load_dparams(module, name, tensor):
    device_mesh = getattr(module, name).device_mesh
    placements = getattr(module, name).placements
    setattr(module, name, nn.Parameter(DTensor.from_local(tensor, device_mesh=device_mesh, placements=placements)))


set_seed(42)
tp_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(tensor_parallel_size=tp_size)
rank = torch.distributed.get_rank()
torch_dtype = torch.float32

config = MoEDolomiteConfig(
    n_embd=1024,
    n_inner=2048,
    num_experts=8,
    num_experts_per_tok=2,
    activation_function="relu",
    add_bias=False,
    embd_pdrop=0.0,
    resid_pdrop=0.0,
)

if rank == 0:
    print(config)

batch_size = 1024
# ones = torch.ones(config.num_experts, device=torch.cuda.current_device(), dtype=torch_dtype)
# eye = torch.eye(config.n_embd, device=torch.cuda.current_device(), dtype=torch_dtype)
# expert_idxs = 1 + torch.arange(config.num_experts, device=torch.cuda.current_device(), dtype=torch_dtype)
# batch_idxs = 1 + torch.arange(batch_size, device=torch.cuda.current_device(), dtype=torch_dtype)
# dim_idxs = 1 + torch.arange(config.n_embd, device=torch.cuda.current_device(), dtype=torch_dtype)

local_moe = ScatterMoE(config, use_padding_free_transformer=True, layer_idx=0)
local_moe = local_moe.to(device=torch.cuda.current_device(), dtype=torch_dtype)
shard_moe = ScatterMoE_TP(config, use_padding_free_transformer=True, sequence_parallel=True, layer_idx=0).to(
    device=torch.cuda.current_device(), dtype=torch_dtype
)
input_tensor = 0.02 * torch.randn(
    batch_size, config.n_embd, device=torch.cuda.current_device(), dtype=torch_dtype, requires_grad=True
)
gate_weight = 0.02 * torch.randn_like(local_moe.gate.weight, requires_grad=True)
c_fc_weight = 0.02 * torch.randn_like(local_moe.c_fc.weight)
c_proj_weight = 0.02 * torch.randn_like(local_moe.c_proj.weight)
grad_tensor = 0.02 * torch.randn(batch_size, config.n_embd, device=torch.cuda.current_device(), dtype=torch_dtype)

torch.distributed.broadcast(input_tensor, 0)
torch.distributed.broadcast(gate_weight, 0)
torch.distributed.broadcast(c_fc_weight, 0)
torch.distributed.broadcast(c_proj_weight, 0)
torch.distributed.broadcast(grad_tensor, 0)


if rank == 0:
    print("Rank", rank)
    print(local_moe)
    print([(n, p.size()) for n, p in local_moe.named_parameters()])
    print(shard_moe)
    print([(n, p.size()) for n, p in local_moe.named_parameters()])

if rank == 0:
    print("Distributing local_moe params...")

params_dict = {"gate.weight": gate_weight, "c_fc.weight": c_fc_weight, "c_proj.weight": c_proj_weight}
local_moe.load_state_dict(params_dict)
torch.distributed.barrier()

if rank == 0:
    print("Distributing shard_moe params...")

# shard_moe.gate.load_state_dict({"weight": gate_weight})
load_dparams(shard_moe.gate, "weight", gate_weight)
load_dparams(shard_moe.c_fc, "weight", c_fc_weight.chunk(tp_size, dim=0)[rank])

# shard_moe.c_proj.load_state_dict({"weight": c_proj_weight.view(c_proj_weight.size(0), c_proj_weight.size(1), tp_size, -1)[:, :, rank]})
load_dparams(
    shard_moe.c_proj,
    "weight",
    c_proj_weight.chunk(tp_size, dim=2)[rank],
)

torch.distributed.barrier()
local_input_tensor = input_tensor
shard_input_tensor = input_tensor.clone().chunk(tp_size, dim=0)[rank]
local_out, local_logits, _ = local_moe(local_input_tensor)
torch.distributed.barrier()
shard_out, shard_logits = shard_moe(shard_input_tensor)

local_input_tensor_grad, local_gate_weight_grad, local_c_fc_grad, local_c_proj_grad = torch.autograd.grad(
    outputs=(local_out),
    inputs=(local_input_tensor, local_moe.gate.weight, local_moe.c_fc.weight, local_moe.c_proj.weight),
    grad_outputs=(grad_tensor,),
)

torch.distributed.barrier()
# print(list(shard_moe.parameters()))
# print(list(local_moe.parameters()))
if rank == 0:
    print("Error:")
    print()
    print("logits:")
for r in range(tp_size):
    if rank == r:
        print("Rank %d:" % r, (local_logits - shard_logits).abs().max())
    torch.distributed.barrier()


if rank == 0:
    print()
    print("out:")

for r in range(tp_size):
    if rank == r:
        print("Rank %d:" % r, (local_out.chunk(tp_size, dim=0)[rank] - shard_out).abs().max())
    torch.distributed.barrier()
shard_input_tensor_grad, shard_gate_weight_grad, shard_c_fc_weight_grad, shard_c_proj_weight_grad = (
    torch.autograd.grad(
        outputs=(shard_out),
        inputs=(shard_input_tensor, shard_moe.gate.weight, shard_moe.c_fc.weight, shard_moe.c_proj.weight),
        grad_outputs=(grad_tensor,),
    )
)

if rank == 0:
    print("gate grads", shard_gate_weight_grad.placements)
    print("c_fc grads", shard_c_fc_weight_grad.placements)
    print("c_proj grads", shard_c_proj_weight_grad.placements)

shard_gate_weight_grad = dtensor_to_tensor(shard_gate_weight_grad, desired_placement=Replicate())
shard_c_fc_weight_grad = dtensor_to_tensor(shard_c_fc_weight_grad, desired_placement=Replicate())
shard_c_proj_weight_grad = dtensor_to_tensor(shard_c_proj_weight_grad, desired_placement=Replicate())

if rank == 0:
    print()
    print("input grad:")
for r in range(tp_size):
    if rank == r:
        diff = (local_input_tensor_grad.chunk(tp_size, dim=0)[rank] - shard_input_tensor_grad).abs()
        print("Rank %d:" % r, diff.max())
    torch.distributed.barrier()

if rank == 0:
    print()
    print("gate grad:")

for r in range(tp_size):
    if rank == r:
        print("Rank %d:" % r, (local_gate_weight_grad - shard_gate_weight_grad).abs().max())
    torch.distributed.barrier()

if rank == 0:
    print()
    print("c_fc grad:")

for r in range(tp_size):
    if rank == r:
        print("Rank %d:" % r, (local_c_fc_grad - shard_c_fc_weight_grad).abs().max())
    torch.distributed.barrier()

if rank == 0:
    print()
    print("c_proj grad:")

for r in range(tp_size):
    if rank == r:
        print("Rank %d:" % r, (local_c_proj_grad - shard_c_proj_weight_grad).abs().max())
    torch.distributed.barrier()

ProcessGroupManager.destroy_process_groups()
