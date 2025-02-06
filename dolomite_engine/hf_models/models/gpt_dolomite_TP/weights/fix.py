import torch

from ....enums import AttentionHeadType
from ....modeling_utils import is_glu
from ...gpt_dolomite import GPTDolomiteConfig


def fix_gpt_dolomite_unsharded_state_dict(
    config: GPTDolomiteConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str = ""
) -> dict:
    state_dict[prefix + "transformer.wte.weight"] = state_dict[prefix + "transformer.wte.weight"][
        : config.vocab_size, :
    ]

    for layer_idx in range(config.n_layer):
        if AttentionHeadType(config.attention_head_type) == AttentionHeadType.mqa:
            q_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.attn.c_attn.q_attn.weight")
            kv_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.attn.c_attn.kv_attn.weight")
            state_dict[f"{prefix}transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.cat([q_attn_w, kv_attn_w])

            if config.add_bias:
                q_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.attn.c_attn.q_attn.bias")
                kv_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.attn.c_attn.kv_attn.bias")
                state_dict[f"{prefix}transformer.h.{layer_idx}.attn.c_attn.bias"] = torch.cat([q_attn_w, kv_attn_w])

        if is_glu(config.activation_function):
            mlp_block_type = config.mlp_blocks[layer_idx]["mlp_block_type"]

            if mlp_block_type == "MLP":
                key = f"{prefix}transformer.h.{layer_idx}.mlp.c_fc.weight"
                weight = state_dict[key].chunk(tensor_parallel_world_size)
                weight = [w.chunk(2) for w in weight]
                w0 = torch.cat([w[0] for w in weight])
                w1 = torch.cat([w[1] for w in weight])
                state_dict[key] = torch.cat([w0, w1])

                if config.add_bias:
                    key = f"{prefix}transformer.h.{layer_idx}.mlp.c_fc.bias"
                    weight = state_dict[key].chunk(tensor_parallel_world_size)
                    weight = [w.chunk(2) for w in weight]
                    w0 = torch.cat([w[0] for w in weight])
                    w1 = torch.cat([w[1] for w in weight])
                    state_dict[key] = torch.cat([w0, w1])
            elif mlp_block_type == "MoE":
                assert not config.add_bias

                key = f"{prefix}transformer.h.{layer_idx}.mlp.c_fc.weight"
                weight = state_dict[key]
                weight = weight.chunk(tensor_parallel_world_size, dim=1)
                weight = [w.chunk(2, dim=1) for w in weight]
                w0 = torch.cat([w[0] for w in weight], dim=1)
                w1 = torch.cat([w[1] for w in weight], dim=1)
                state_dict[key] = torch.cat([w0, w1], dim=1)
            else:
                raise ValueError(f"unexpected mlp_block_type ({mlp_block_type})")

    return state_dict
