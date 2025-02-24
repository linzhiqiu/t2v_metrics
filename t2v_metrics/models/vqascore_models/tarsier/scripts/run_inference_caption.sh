#!/bin/bash

# Copy and Modified on: https://github.com/LLaVA-VL/LLaVA-NeXT/blob/video_inference/scripts/video/eval/video_detail_description_eval_shard.sh

# 

model_name_or_path=$1
input_file=$2
output_dir=$3
CHUNKS=1
resume=True

mkdir $output_dir

echo "Using $CHUNKS GPUs"

# Assuming GPULIST is a bash array containing your GPUs
# GPULIST=(0 1 2 3 4 5 6 7)
GPULIST=(0)

# Get the number of GPUs
NUM_GPUS=${#GPULIST[@]}

# Calculate GPUs per chunk
GPUS_PER_CHUNK=$((NUM_GPUS / CHUNKS))


for IDX in $(seq 1 $CHUNKS); do
    START=$(((IDX-1) * GPUS_PER_CHUNK))
    LENGTH=$GPUS_PER_CHUNK # Length for slicing, not the end index
    
    CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})
    
    # Convert the chunk GPUs array to a comma-separated string
    CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")

    ALL_GPUS_FREE=0
    while [ $ALL_GPUS_FREE -eq 0 ]; do
        ALL_GPUS_FREE=1  # Assume all GPUs are free initially
        
        for GPU_ID in $CHUNK_GPUS; do
            MEM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | tr -d '[:space:]')
            
            # Assuming a GPU is considered free if its memory usage is less than 100 MiB
            if [ "$MEM_USAGE" -ge 100 ]; then
                ALL_GPUS_FREE=0
                echo "GPU $GPU_ID is in use. Memory used: ${MEM_USAGE}MiB."
                break  # Exit the loop early as we found a GPU that is not free
            fi
        done
        
        if [ $ALL_GPUS_FREE -eq 0 ]; then
            echo "Not all GPUs in chunk are free. Checking again in 10 seconds..."
            sleep 10
        fi
    done
    
    echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
    CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 -m tasks.inference_caption \
        --model_name_or_path $model_name_or_path \
        --config "configs/tarser2_default_config.yaml" \
        --max_new_tokens 512 \
        --top_p 1 \
        --temperature 0 \
        --input_file $input_file \
        --output_dir $output_dir \
        --output_name predictions \
        --max_n_samples_per_benchmark -1 \
        --resume $resume \
        --num_chunks $CHUNKS \
        --chunk_idx $(($IDX - 1)) > $output_dir/run_$IDX.log 2>&1 &

done

wait

# python3 -m evaluation.evaluate \
#     --pred_file $output_dir \
#     --benchmarks $benchmarks
