import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dolomite_engine.hf_models import import_from_huggingface_llama


# load our and hf implementation
import_from_huggingface_llama("meta-llama/Llama-2-7b-chat-hf", "tmp")
model_megatron = AutoModelForCausalLM.from_pretrained("tmp", torch_dtype=torch.float16, device_map=0).cuda()

model_hf = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map=0
).cuda()

# disable dropout
model_hf.eval()
model_megatron.eval()

# print architecture
print(model_hf)
print(model_megatron)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# some dummy input
x = tokenizer("def generate():", return_tensors="pt")
x = {i: x[i].cuda() for i in x}

# happy generation
y_hf = model_hf.generate(**x, max_new_tokens=100, do_sample=False)
y_megatron = model_megatron.generate(**x, max_new_tokens=100, do_sample=False)

print(tokenizer.batch_decode(y_hf)[0])
print("-" * 50)
print(tokenizer.batch_decode(y_megatron)[0])
