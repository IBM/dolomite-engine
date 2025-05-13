#!/bin/bash
#SBATCH --partition=gh200
#SBATCH --nodes=4
#SBATCH --job-name=torchrun-test
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

. ~/.bashrc

echo "hostname = $(hostname)"

export OMP_NUM_THREADS=72
export HF_HOME="/mnt/vast/users/${USER}/.cache/huggingface/"

CONTAINER="/mnt/vast/squash/dolomite.sqsh"
MOUNTS="/mnt/home/${USER}:/mnt/home/${USER},/mnt/vast/:/mnt/vast"

SRUN_ARGS=" --kill-on-bad-exit=1 --container-image=${CONTAINER} --container-mounts=${MOUNTS} "

echo "srun args: ${SRUN_ARGS}"

cd ${HOME}/dolomite-engine

#export MASTER_ADDR=$(echo $SLURM_JOB_NODELIST | head -n 1 )
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1 |  hostname --ip-address)
export CUDA_VISIBLE_DEVICES=0 # only 1 gpu(0) on gh200 node 
export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR=$(hostname) # use IP addr of headnode for communication
export MASTER_PORT=28444
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# print some stuff for debug 
echo "Using $MASTER_ADDR"
echo "Using Nnodes: $NNODES"
echo "Using SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "Using node List : $SLURM_JOB_NODELIST"

echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

#add some new Torch NCCL directives for debugging 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #for newer versions of pytorch If set to 1, aborting NCCL communicator and tearing down process upon error.
export PYTHONUNBUFFERED=TRUE
export WANDB_MODE=offline #leverage wnadb for run stats, but do not upload to the cloud 
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_CUDA_ARCH_LIST="9.0a" #Grace hopper reports as 9.0a, not 9.0 ?GH/GB200

export DIST_ARGS="--nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    "
echo "torch run args= ${DIST_ARGS}"
echo " "
echo "srun ${SRUN_ARGS} torchrun $DIST_ARGS -m dolomite_engine.pretrain  --config $config >>$consolelog 2>&1 "
echo " "

srun ${SRUN_ARGS} torchrun $DIST_ARGS ~/ai-coreweave/run/test/torch-test.py ${MASTER_ADDR}  

#alternate method is to use a second script to run the code 
#srun ${SRUN_ARGS} ~/dolomite-engine/scripts/coreweave/torchrun-test-srun.sh

exit
