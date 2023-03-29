[paper]: https://img.shields.io/static/v1?label=&message=Paper&color=blueviolet


# Distributed finetuning
This repository is meant for finetuning large language models (of any scale) using [DeepSpeed](https://github.com/microsoft/DeepSpeed). Right now 2 main class of models from [HuggingFace](https://huggingface.co/docs/transformers/index) are supported:

1. decoder models like GPT2, OPT, BLOOM etc
1. encoder-decoder models like T5, BART etc

Please note that this repository doesn't support Tensor Parallel or Pipeline Parallel (yet ;p). For distributed training, we use ZeRO-DP from DeepSpeed.

The repository supports models of any scale. But, I don't recommend training models larger than 20 Billion using this repository as using ZeRO-DP is not the most efficient method of scaling to hyper-large models.

For pre-training from scratch or finetuning on huge amount of data, check out [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed).

# Usage

Please note that the training scripts need to be launched on all nodes parallely for multinode training. For training and inference, take a look at [scripts](scripts).

### Training on CCC
For example, for using 2 nodes with 4 GPUs each (8 GPUs) on CCC, it can be done using:
```shell
jbsub -q x86_24h -cores 2x8+4 -mem 128G -require a100_80gb -err err.log -out out.log blaunch.sh sh scripts/full_finetuning/train_ccc.sh
```
Note that the `blaunch.sh` script here (provided by CCC) executes the command `sh scripts/prompt_tuning/train_ccc.sh` on both the nodes.

### Training on DIPC Openshift cluster
Take a look at https://github.ibm.com/Mayank-Mishra1/dipc-openshift for launching jobs.

## Arguments for training
The training script currently supports the following arguments:
```shell
model:
  --model_name MODEL_NAME
                        model name on huggingface hub
  --model_class {<class 'transformers.models.auto.modeling_auto.AutoModelForCausalLM'>,<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>}
                        model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM

checkpointing:
  --save_path SAVE_PATH
                        path to save checkpoints

dataset:
  --data_sampling_proportion DATA_SAMPLING_PROPORTION [DATA_SAMPLING_PROPORTION ...]
                        sampling proportion for the datasets
  --data_path DATA_PATH [DATA_PATH ...]
                        list of datapaths
  --data_class DATA_CLASS [DATA_CLASS ...]
                        list of dataclasses to use
  --data_config DATA_CONFIG [DATA_CONFIG ...]
                        list of data configs to use
  --input_format INPUT_FORMAT [INPUT_FORMAT ...]
                        list of format of input in examples in the datasets
  --output_format OUTPUT_FORMAT [OUTPUT_FORMAT ...]
                        list of format of output in examples in the datasets
  --max_input_tokens MAX_INPUT_TOKENS [MAX_INPUT_TOKENS ...]
                        max length for input
  --max_output_tokens MAX_OUTPUT_TOKENS [MAX_OUTPUT_TOKENS ...]
                        max length for output

miscellaneous:
  --seed SEED           random seed
  --dtype {torch.float32,torch.float16,torch.bfloat16}
                        dtype to use for training / inference

prompt tuning initialization:
  --prompt_tuning_init PROMPT_TUNING_INIT
  --prompt_tuning_init_text PROMPT_TUNING_INIT_TEXT
  --num_virtual_tokens NUM_VIRTUAL_TOKENS

training inference:
  --training_inference_type {TrainingInferenceType.full_finetuning,TrainingInferenceType.prompt_tuning}
                        type of tuning, full finetuning or PEFT

training:
  --num_training_steps NUM_TRAINING_STEPS
                        number of training steps
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        gradient accumulation steps
  --eval_and_save_interval EVAL_AND_SAVE_INTERVAL
                        interval for evaluation and checkpointing
  --batch_size_per_gpu BATCH_SIZE_PER_GPU
                        batch size per GPU for ZeRO-DP
  --no_eval             avoid evaluating val dataset during training

parallelism:
  --stage STAGE         deepspeed ZeRO stage
  --overlap_comm        overlap communication with computation
  --contiguous_gradients
                        use contiguous buffers for gradients, requires more memory if enabled
  --cpu_offload         train with CPU offloading to save GPU memory

logging:
  --logdir LOGDIR       logging directory for experiments

aim:
  --aim_repo AIM_REPO   aim repo, experiment logs are saved here
  --experiment_name EXPERIMENT_NAME
                        name of the experiment

optimizer and scheduler:
  --learning_rate LEARNING_RATE
  --weight_decay WEIGHT_DECAY
  --beta1 BETA1
  --beta2 BETA2
  --eps EPS
  --warmup_steps WARMUP_STEPS
  --lr_schedule {LearningRateScheduler.linear,LearningRateScheduler.cosine}
                        learning rate schedule

debug:
  --steps_per_print STEPS_PER_PRINT
                        steps per print
```

## Arguments for inference
```shell
model:
  --model_name MODEL_NAME
                        model name on huggingface hub
  --model_class {<class 'transformers.models.auto.modeling_auto.AutoModelForCausalLM'>,<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>}
                        model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM

checkpointing:
  --load_path LOAD_PATH
                        path to load checkpoints

dataset:
  --data_sampling_proportion DATA_SAMPLING_PROPORTION [DATA_SAMPLING_PROPORTION ...]
                        sampling proportion for the datasets
  --data_path DATA_PATH [DATA_PATH ...]
                        list of datapaths
  --data_class DATA_CLASS [DATA_CLASS ...]
                        list of dataclasses to use
  --data_config DATA_CONFIG [DATA_CONFIG ...]
                        list of data configs to use
  --input_format INPUT_FORMAT [INPUT_FORMAT ...]
                        list of format of input in examples in the datasets
  --output_format OUTPUT_FORMAT [OUTPUT_FORMAT ...]
                        list of format of output in examples in the datasets
  --max_input_tokens MAX_INPUT_TOKENS [MAX_INPUT_TOKENS ...]
                        max length for input
  --max_output_tokens MAX_OUTPUT_TOKENS [MAX_OUTPUT_TOKENS ...]
                        max length for output

miscellaneous:
  --seed SEED           random seed
  --dtype {torch.float32,torch.float16,torch.bfloat16}
                        dtype to use for training / inference

prompt tuning initialization:
  --prompt_tuning_init PROMPT_TUNING_INIT
  --prompt_tuning_init_text PROMPT_TUNING_INIT_TEXT
  --num_virtual_tokens NUM_VIRTUAL_TOKENS

training inference:
  --training_inference_type {TrainingInferenceType.full_finetuning,TrainingInferenceType.prompt_tuning}
                        type of tuning, full finetuning or PEFT

inference:
  --batch_size BATCH_SIZE
                        batch size
  --do_sample           sample or greedy
  --max_new_tokens MAX_NEW_TOKENS
                        max new tokens
  --temperature TEMPERATURE
                        temperature
  --top_k TOP_K         top k
  --top_p TOP_P         top p

output:
  --output_file OUTPUT_FILE
                        output file
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

# Distributed training research papers

1. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
1. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
1. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
1. [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032)
1. [Meet Gemini:The Heterogeneous Memory Manager of Colossal-AI](https://colossalai.org/docs/advanced_tutorials/meet_gemini/)
