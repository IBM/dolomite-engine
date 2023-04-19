# number of nodes is stored in WORLD_SIZE by kubeflow
# node rank is stored in RANK by kubeflow

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=101 \
    --rdzv_endpoint=$HOSTNAME:$MASTER_PORT \
    -m src.train \
    --config configs/sst2/full_finetuning-training.json
