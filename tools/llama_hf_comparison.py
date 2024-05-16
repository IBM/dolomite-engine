import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dolomite_engine.hf_models import import_from_huggingface_llama


# load our and hf implementation
import_from_huggingface_llama("meta-llama/Llama-2-7b-chat-hf", "tmp")
dolomite_model = AutoModelForCausalLM.from_pretrained("tmp", torch_dtype=torch.float16, device_map=0).cuda()

hf_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map=0
).cuda()

# disable dropout
hf_model.eval()
dolomite_model.eval()

# print architecture
print(hf_model)
print(dolomite_model)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# some dummy input
x = tokenizer("def generate():", return_tensors="pt")
x = {i: x[i].cuda() for i in x}

# happy generation
y_hf = hf_model.generate(**x, max_new_tokens=100, do_sample=False)
y_dolomite = dolomite_model.generate(**x, max_new_tokens=100, do_sample=False)

print(tokenizer.batch_decode(y_hf)[0])
print("-" * 50)
print(tokenizer.batch_decode(y_dolomite)[0])
