<h1 align="center" style="font-size: 50px;">Dolomite Engine</h1>

<p align="center">
  <img src="assets/dolomite.jpeg" width="300px" height="300px">
</p>

<!-- Topic -->
[Efficient Training]: https://img.shields.io/static/v1?label=&message=Efficient%20Training&color=blueviolet
[Efficient Inference]: https://img.shields.io/static/v1?label=&message=Efficient%20Inference&color=blueviolet
[Instruction Finetuning]: https://img.shields.io/static/v1?label=&message=Instruction%20Finetuning&color=blueviolet
[Mixture of Experts]: https://img.shields.io/static/v1?label=&message=Mixture%20of%20Experts&color=blueviolet
[Model Architecture]: https://img.shields.io/static/v1?label=&message=Model%20Architecture&color=blueviolet

# Introduction
This repository contains code used for pretraining and finetuning IBM's Granite models. It also includes the following key innovations on model architectures, finetuning methods, systems optimizations:
1. [Saving Memory Using Padding-Free Transformer Layers during Finetuning](https://huggingface.co/blog/mayank-mishra/padding-free-transformer)  
_Mayank Mishra_  
![image][Efficient Training]
1. [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981)  
_William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, Jonathan Ragan Kelly_  
![image][Efficient Inference] ![image][Model Architecture]
1. [Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models](https://arxiv.org/abs/2404.05567)  
_Bowen Pan, Yikang Shen, Haokun Liu, Mayank Mishra, Gaoyuan Zhang, Aude Oliva, Colin Raffel, Rameswar Panda_  
![image][Mixture of Experts] ![image][Efficient Inference] ![image][Model Architecture]
1. [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)  
_Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein_  
![image][Instruction Finetuning]
1. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)  
_Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim_  
![image][Model Architecture] ![image][Efficient Training] ![image][Efficient Inference]
1. [Scattered Mixture-of-Experts Implementation](https://arxiv.org/abs/2403.08245)  
_Shawn Tan, Yikang Shen, Rameswar Panda, Aaron Courville_  
![image][Mixture of Experts] ![image][Efficient Training] ![image][Efficient Inference]

# Getting Started
Run `make install` to install the requirements for this repository. You might need to install `flash-attn`.

# Distributed finetuning
This repository is meant for finetuning large language models (of any scale) using multiple backends. The following backends are currently supported:

1. [FSDP](https://pytorch.org/docs/stable/fsdp.html)
1. [DeepSpeed](https://github.com/microsoft/DeepSpeed)

The repository currently only supports generative models but can be easily extended to non-generative models if needed. 2 main class of models from [HuggingFace](https://huggingface.co/docs/transformers/index) are supported:

1. decoder models (`AutoModelForCausalLM`) like Granite, Llama, BLOOM etc
1. encoder-decoder models (`AutoModelForSeq2SeqLM`) like T5, BART etc

Please note that this repository doesn't support Tensor Parallel or Pipeline Parallel (yet :wink:).

# HuggingFace compatible custom models
This repository works with all HuggingFace models (text-to-text only for the moment) out-of-the-box. The checkpoints have to be in safetensors format, if not you can check `tools/pt_to_safetensors.py`. If your model_type is `gpt_megatron` just change it to `gpt_dolomite`.

> [!TIP]
> You might be able to enjoy additional memory and computation savings when finetuning your models using the [padding free transformers optimization](https://huggingface.co/blog/mayank-mishra/padding-free-transformer). This optimization is currently only supported for decoder models and requires converting your model (say LLama-3 for example) to a [custom class](dolomite_engine/hf_models/models/gpt_dolomite/) implemented in this repo. This is completely optional and not required for finetuning. The conversion can be achieved as follows:
```python
from dolomite_engine.hf_models import import_from_huggingface

import_from_huggingface(
    pretrained_model_name_or_path="ibm-granite/granite-3b-code-base",
    save_path="dolomite_compatible_model"
)
```
Once done training, you can convert the model back to the HF class as:
```python
from dolomite_engine.hf_models import export_to_huggingface

export_to_huggingface(
    pretrained_model_name_or_path="trained_checkpoint",
    save_path="hf_compatible_model",
    model_type="llama",
)
```

If you are interested in using this optimization outside this repo for some reason, you can do as follows:
```python
import torch
from dolomite_engine.hf_models import GPTDolomiteForCausalLM


# we need unpadded lists here for avoiding any useless computations on pad tokens
# this is a bit different from the standard transformer which takes in tensors and an attention mask
# if you turn off padding free transformers, you can use the tensor inputs with this class too
input_ids = [[1, 2, 3, 4, 5, 0], [6, 7, 8, 0]]
labels = [[-100, -100, -100, 4, 5, 0], [-100, -100, 8, 0]]

# this will throw a warning saying that the model is of gpt_bigcode class
# ignore the warning
model = GPTDolomiteForCausalLM.from_pretrained(
    <model_path>,
    attn_implementation="flash_attention_2"
    use_padding_free_transformer=True,
).cuda()

loss = model(
    input_ids=input_ids,
    labels=labels,
).loss
```

Note that padding free transformers doesn't support generation and thus for running generation on the model, you will need to load the model without padding-free transformers.

# Usage
The typical training workflow looks like:
1. [Pretraining](scripts/pretrain.sh) or [Finetuning](scripts/finetune.sh): This is the actual training process
```shell
# for finetuning
sh scripts/finetune.sh configs/sst2/training.yml
```
```shell
# for pretraining
sh scripts/pretrain.sh configs/pretraining-examples/pretrain-1.yml
```

2. [Inference](scripts/generate.sh): Run inference on the trained models or the un-trained model
```shell
sh scripts/generate.sh configs/sst2/inference.yml
```

3. [Unshard the checkpoint](scripts/unshard.sh): This is used to unshard the model to a safetensors checkpoint since dolomite-engine saves a sharded model during training
```shell
sh scripts/unshard.sh configs/sst2/unshard.yml
```

## Using custom datasets
The data directory should obey the following structure:
```text
📦data
 ┣ 📂train
 ┃ ┣ 📜filename1.jsonl
 ┃ ┣ 📜filename2.jsonl
 ┃ ┗ 📜filename3.jsonl
 ┗ 📂val
 ┃ ┣ 📜filename1.jsonl
 ┃ ┣ 📜filename2.jsonl
 ┃ ┣ 📜filename3.jsonl
 ┣ 📂test
 ┃ ┣ 📜filename1.jsonl
 ┃ ┣ 📜filename2.jsonl
 ┃ ┣ 📜filename3.jsonl
```
Filenames can be anything as long as there are no whitespaces in them. Each line in each file should be a json (jsonlines file format) with the entries looking like:
```json
{"input": "The movie sucks", "output": "negative"}
{"input": "The movie was awesome", "output": "positive"}
```
Note for the test set, only `input` field is needed in the json instances in each line. `output` field is not needed.  

All the files in each directory are concatenated to form the respective split.  

If you need reformatting of the examples, you can use `input_format` and `output_format` arguments. For example `input_format = 'Classify the sentiment of the sentence:\n__input__\nSentiment:'` and `output_format = ' __output__'` reformats the input and output examples to:
```text
INPUT:
Classify the sentiment of the sentence:
The movie sucks
Sentiment:

OUTPUT:
 negative
```
If you don't need any reformatting, leave the arguments `input_format` and `output_format` to their default values `__input__` and `__output__` respectively.  

Please note that the user is expected to provide this at both training and inference time.  

Try not to have trailing spaces in `input_format`, if you need a space between input and output, the space should be part of the `output_format` as in the above example.

> [!TIP]
> Alternatively, you can also add your own dataset class in the repository if you don't want to use the jsonlines format or need custom logic to load your own dataset.

Currently, the repo has following implemented dataclasses:
```text
AlpacaDataset
DebugDataset
DollyDataset
HuggingFaceDataset
SlimOrcaDataset
SST2Dataset
```

## Using Megatron Dataset outside of this repository
This repository implements the dataloader from Megatron-LM for efficient pretraining. If for some reason you need to use that dataloader outside this repository, take a look at [this example](tools/megatron_dataset/megatron_dataloader.py).

## Supported optimizers
We support all of the following optimizers. The default optimizer is `TorchAdamW`. Note that using the [DeepSpeed](https://github.com/microsoft/DeepSpeed) or [Apex](https://github.com/NVIDIA/apex) optimizers will require installing the respective pip package.

```python
# https://nvidia.github.io/apex/optimizers.html
from apex.optimizers import FusedAdam as ApexFusedAdam
from apex.optimizers import FusedLAMB as ApexFusedLAMB
from apex.optimizers import FusedNovoGrad as ApexFusedNovoGrad
from apex.optimizers import FusedSGD as ApexFusedSGD

# https://deepspeed.readthedocs.io/en/latest/optimizers.html
from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam as DeepSpeedFusedAdam
from deepspeed.ops.lamb import FusedLamb as DeepSpeedFusedLAMB
from deepspeed.runtime.fp16.onebit import OnebitAdam as DeepSpeedOnebitAdam
from deepspeed.runtime.fp16.onebit import OnebitLamb as DeepSpeedOnebitLAMB
from deepspeed.runtime.fp16.onebit import ZeroOneAdam as DeepSpeedZeroOneAdam

# https://pytorch.org/docs/stable/optim.html
from torch.optim.adadelta import Adadelta as TorchAdadelta
from torch.optim.adagrad import Adagrad as TorchAdagrad
from torch.optim.adam import Adam as TorchAdam
from torch.optim.adamax import Adamax as TorchAdamax
from torch.optim.adamw import AdamW as TorchAdamW
from torch.optim.asgd import ASGD as TorchASGD
from torch.optim.lbfgs import LBFGS as TorchLBFGS
from torch.optim.nadam import NAdam as TorchNAdam
from torch.optim.radam import RAdam as TorchRAdam
from torch.optim.rmsprop import RMSprop as TorchRMSprop
from torch.optim.rprop import Rprop as TorchRprop
from torch.optim.sgd import SGD as TorchSGD
```

# Citation
If you find this repository useful, please consider citing it in your research:
```bibtex
@software{Mishra_Dolomite_Engine_A_2024,
    author = {Mishra, Mayank},
    month = jun,
    title = {{Dolomite Engine: A Hyper-Optimized Library for Pretraining and Finetuning}},
    url = {https://github.com/ibm-granite/dolomite-engine},
    year = {2024}
}
```
