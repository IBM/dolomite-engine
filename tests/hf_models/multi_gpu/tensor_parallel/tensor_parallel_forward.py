import argparse
import os

import torch
import torch.distributed
from transformers import set_seed

from dolomite_engine.hf_models import AttentionHeadType, GPTDolomiteConfig, MoEDolomiteConfig, get_model_parallel_class
from dolomite_engine.utils import ProcessGroupManager, SafeTensorsWeightsManager, string_to_torch_dtype

from ...test_common import TestCommons


parser = argparse.ArgumentParser()
parser.add_argument("--attention-head-type", type=str)
parser.add_argument("--position-embedding-type", type=str)
parser.add_argument("--attention-implementation", type=str)
parser.add_argument("--torch-dtype", type=str)
parser.add_argument("--tmp-path", type=str)
parser.add_argument("--tensor-parallel-word-embeddings", action="store_true")
parser.add_argument("--use-padding-free-transformer", action="store_true")
parser.add_argument("--sequence-parallel", action="store_true")
parser.add_argument("--model-type", type=str)
args = parser.parse_args()

set_seed(42)

ProcessGroupManager(tensor_parallel_world_size=int(os.getenv("WORLD_SIZE")))

torch_dtype = string_to_torch_dtype(args.torch_dtype)

num_key_value_heads = None
if AttentionHeadType(args.attention_head_type) == AttentionHeadType.gqa:
    num_key_value_heads = 8

kwargs = {}
if args.model_type == GPTDolomiteConfig.model_type:
    config = GPTDolomiteConfig(
        attention_head_type=args.attention_head_type,
        n_layer=1,
        position_embedding_type=args.position_embedding_type,
        num_key_value_heads=num_key_value_heads,
        add_bias=False,
        n_embd=128,
        n_head=16,
    )
elif args.model_type == MoEDolomiteConfig.model_type:
    config = MoEDolomiteConfig(
        attention_head_type=args.attention_head_type,
        n_layer=1,
        position_embedding_type="learned_absolute",
        num_key_value_heads=num_key_value_heads,
        add_bias=False,
        n_embd=128,
        n_head=16,
    )
    kwargs["moe_implementation"] = "scattermoe"


if torch.distributed.get_rank() == 0:
    with torch.device("meta"):
        model = TestCommons.from_config(None, config, attn_implementation=args.attention_implementation)

    model = model.to_empty(device=torch.cuda.current_device())
    for _, param in model.named_parameters():
        param.data.normal_(0, 0.0125)

    model.eval()

    model.save_pretrained(args.tmp_path, safe_serialization=True)
    model = model.to(torch_dtype)

torch.distributed.barrier()

# use dummy tensors to avoid initializing model here
with torch.device("meta"):
    # try sharding vocab matrices if really struggling for memory

    model_tp = get_model_parallel_class(args.model_type)._from_config(
        config,
        tensor_parallel_word_embeddings=args.tensor_parallel_word_embeddings,
        attn_implementation=args.attention_implementation,
        use_padding_free_transformer=args.use_padding_free_transformer,
        sequence_parallel=args.sequence_parallel,
        **kwargs,
    )

# copy to device without copying storage
model_tp = model_tp.to_empty(device=torch.cuda.current_device())

# load weights into tensor parallel model using SafeTensorsWeightsManager class
# this avoids loading multiple copies of the parameters in CPU memory
model_tp.load_from_safetensors_weights_manager(SafeTensorsWeightsManager(args.tmp_path))

# set model to eval mode
model_tp = model_tp.to(torch_dtype)
model_tp.eval()

set_seed(42)

batch_size = 4
sequence_length = 512

input_ids = torch.randint(
    0, 50255, (batch_size, sequence_length), device=torch.cuda.current_device(), requires_grad=False
)
labels = torch.randint(
    0, 50255, (batch_size, sequence_length), device=torch.cuda.current_device(), requires_grad=False
)

if args.use_padding_free_transformer:
    cu_seqlens = torch.arange(
        0, input_ids.numel() + 1, sequence_length, dtype=torch.int32, device=torch.cuda.current_device()
    )
    max_seqlen = torch.tensor(sequence_length, device=torch.cuda.current_device())
    position_ids = torch.arange(0, sequence_length, 1, device=torch.cuda.current_device()).repeat(batch_size)

    output_tp = model_tp(
        input_ids=input_ids.view(-1),
        labels=labels.view(-1),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        position_ids=position_ids,
    )
else:
    output_tp = model_tp(input_ids=input_ids, labels=labels)

loss_tp = output_tp[0]
logits_tp = output_tp[1]

if args.tensor_parallel_word_embeddings:
    logits_tp = logits_tp[..., : config.vocab_size]

if torch.distributed.get_rank() == 0:
    output = model(input_ids=input_ids, labels=labels)
    loss = output[0]
    logits = output[1]

    if args.use_padding_free_transformer:
        logits_tp = logits_tp.reshape(batch_size, sequence_length, -1)

    error = (logits - logits_tp).abs().max()
    assert error < 5e-4, "logits don't match for normal and tensor parallel model"

    error = (loss - loss_tp).abs().max()
    assert error < 3e-6, "losses don't match for normal and tensor parallel model"

ProcessGroupManager.destroy_process_groups()
