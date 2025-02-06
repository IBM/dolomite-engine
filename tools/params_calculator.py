import torch
from transformers import AutoModelForCausalLM

from dolomite_engine.hf_models import GPTDolomiteConfig


config = GPTDolomiteConfig(
    vocab_size=65024,
    max_position_embeddings=4096,
    hidden_size=8192,
    num_layers=72,
    n_head=64,
    num_key_value_heads=8,
    n_inner=21888,
    position_embedding_type="rope",
    activation_function="swiglu",
    normalization_function="rmsnorm",
    attention_head_type="gqa",
)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

num_parameters = 0
for param_name, param in model.named_parameters():
    num_parameters += param.numel()

print("\ntotal", f"{num_parameters:,}")
