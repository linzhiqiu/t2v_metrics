#!/bin/bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

# Ensure PYTHONPATH is set to the project root (InternVideo2/)
export PYTHONPATH=$(realpath $(dirname "$0")/../../../../../)

echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
NUM_GPUS=1
NUM_CPU=16

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0  # Change if using multiple GPUs
export MASTER_ADDR=localhost  # If running on a single node
export WORLD_SIZE=1
export RANK=0

# Run training with torchrun (ensure the correct module path)
torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT} \
    -m multi_modality.tasks.pretrain \
    --config $(dirname $0)/config_msrvtt.py \
    --output_dir ${OUTPUT_DIR} \
    --evaluate True \
    --pretrained_path '/data3/zhiqiul/t2i_metrics/hf_cache/internvideo2_1b_stage2.pth'
