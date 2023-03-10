# number of nodes is stored in WORLD_SIZE by kubeflow
# node rank is stored in RANK by kubeflow

NUM_GPUS=8
MASTER_ADDRESS=pytorchjob-mayank-debug-master-0

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=101 \
    --rdzv_endpoint=$MASTER_ADDRESS:$MASTER_PORT \
    -m src.train \
    --model_name bigscience/bloom-560m \
    --model_class AutoModelForCausalLM \
    --training_inference_type prompt_tuning \
    --prompt_tuning_init TEXT \
    --prompt_tuning_init_text $'Classify the sentiment of the sentence:' \
    --num_virtual_tokens 8 \
    --data_class JSONLinesDataset \
    --input_format $'\n__input__\nSentiment:' \
    --output_format $' __output__' \
    --max_input_tokens 1024 \
    --max_output_tokens 128 \
    --experiment_name sst2-bloom-560m-prompt_tuning \
    --save_path checkpoints/prompt_tuning \
    --num_training_steps 4000 \
    --eval_and_save_interval 500 \
    --batch_size_per_gpu 8 \
    --dtype bfloat16 \
    --learning_rate 1e-5 \
    --lr_schedule cosine \
    --data_path data
