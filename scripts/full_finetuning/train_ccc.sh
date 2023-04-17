# export PYTHONFAULTHANDLER=1
# export NCCL_DEBUG="INFO"
# export NCCL_DEBUG_FILE="$LOG_PATH/NCCL_DEBUG.%h.%p.txt"
# export NCCL_TOPO_DUMP_FILE="$LOG_PATH/NCCL_TOP.%h.xml"
export NCCL_SOCKET_IFNAME="ib,bond"
export NCCL_IB_CUDA_SUPPORT=1

MASTER_ADDRESS=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=5${LSB_JOBID: -5:-1}
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=101 \
    --rdzv_endpoint=$MASTER_ADDRESS:$MASTER_PORT \
    -m src.train \
    --model_name bigscience/bloom-560m \
    --model_class AutoModelForCausalLM \
    --training_inference_type full_finetuning \
    --dataset_configs_json configs/sst2-full_finetuning.json \
    --experiment_name sst2-bloom-560m-full_finetuning \
    --save_path checkpoints/full_finetuning \
    --num_training_steps 4000 \
    --eval_and_save_interval 500 \
    --batch_size_per_gpu 8 \
    --dtype bfloat16 \
    --learning_rate 1e-5 \
    --lr_schedule cosine
