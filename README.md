<h1 align="center" style="font-size: 50px;">Dolomite Engine</h1>

<p align="center">
  <img src="assets/dolomite.webp" width="300px" height="300px">
  <br>
   Image of Dolomite generated using DALL-E
</p>


# Distributed finetuning
This repository is meant for finetuning large language models (of any scale) using multiple backends. The following backends are currently supported:

1. [DeepSpeed](https://github.com/microsoft/DeepSpeed)
2. [FSDP](https://pytorch.org/docs/stable/fsdp.html)

The repository currently only supports generative models but can be easily extended to non-generative models if needed. 2 main class of models from [HuggingFace](https://huggingface.co/docs/transformers/index) are supported:

1. decoder models like GPT2, OPT, BLOOM etc
1. encoder-decoder models like T5, BART etc

Please note that this repository doesn't support Tensor Parallel or Pipeline Parallel (yet ;p). For distributed training, we use ZeRO-DP from DeepSpeed.

The repository supports models of any scale. But, I don't recommend training models larger than 20 Billion using this repository as using ZeRO-DP is not the most efficient method of scaling to hyper-large models.

For pre-training from scratch or finetuning on huge amount of data, check out [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed).

# Usage

Please note that the training scripts need to be launched on all nodes parallely for multinode training. For training and inference, take a look at [scripts](scripts/) to train on sst2.

### Training on CCC
For example, for using 2 nodes with 4 GPUs each (8 GPUs) on CCC, it can be done using:
```shell
jbsub -q x86_24h -cores 2x8+4 -mem 128G -require a100_80gb -err err.log -out out.log blaunch.sh sh scripts/train_ccc.sh
```
Note that the `blaunch.sh` script here (provided by CCC) executes the command `sh scripts/train_ccc.sh` on both the nodes.

### Training on Vela cluster
```shell
helm template -f scripts/train_vela.yaml chart | tee appwrapper.yaml | oc create -f -
```

## To use Flash Attention for IBM models
Install the latest release of https://github.ibm.com/ai-models-architectures/ibm-models. Simple pip install will work.
Use this [config](configs/8k/).

## Supported optimizers
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

## Dataset
The data directory should obey the following structure:
```text
📦data
 ┣ 📂train
 ┃ ┣ 📜filename1.jsonl
 ┃ ┣ 📜filename2.jsonl
 ┃ ┗ 📜filename3.jsonl
 ┗ 📂val
 ┃ ┣ 📜filename1.jsonl
 ┣ 📂test
 ┃ ┣ 📜filename1.jsonl
 ┃ ┣ 📜filename2.jsonl
```
Filenames can be anything as long as there are no whitespaces in them. Each line in each file should be a json (jsonlines file format) with the entries looking like:
```json
{"input": "The movie sucks", "output": "negative"}
{"input": "The movie was awesome", "output": "positive"}
```
Note for the test set, only `"input"` field is needed in the json instances in each line. `"output"` field is not needed. \
All the files in each directory are concatenated to form the respective split. \
If you need reformatting of the examples, you can use `--input_format` and `--output_format` arguments. For example `--input_format = 'Classify the sentiment of the sentence:\n__input__\nSentiment:'` and `--output_format = ' __output__'` reformats the input and output examples to:
```text
INPUT:
Classify the sentiment of the sentence:
The movie sucks
Sentiment:

OUTPUT:
 negative
```
If you don't need any reformatting, leave the arguments `--input_format` and `--output_format` to their default values `__input__` and `__output__` respectively. \
Please note that the user is expected to provide this at both training and inference time. \
Try not to have trailing spaces in `input_format`, if you need a space between input and output, the space should be part of the `output_format`.

# Tracking experiments
The repository also supports [aim](https://github.com/aimhubio/aim) based experiment tracking. The default `--aim_repo` for training is `./aim_repo`. To change this, specify `--aim_repo <PATH_TO_AIM_REPO>` arg during training. For viewing the aim dashboard, simply run
```shell
aim up --host <HOST_ADDRESS> --port <HOST_PORT> --repo <PATH_TO_AIM_REPO>
```
This brings up an awesome UI like this:
![image](https://user-images.githubusercontent.com/13848158/136374529-af267918-5dc6-4a4e-8ed2-f6333a332f96.gif)

# Contribution
Feel free to open any issues and open pull requests to contribute code :)
