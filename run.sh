#!/bin/bash

# List of model names
MODELS=(
    'internvl2.5-26b'
    'internvideo2-chat-8b-internlm'
    "llava-onevision-qwen2-7b-ov"
    'qwen2.5-vl-7b'
    'tarsier-recap-7b'
    'llava-video-7B'
    'mplug-owl3-7b'
    'internlmxcomposer25-7b'
)

# Set batch size and question/answer templates
BATCH_SIZE=64
QUESTION="What is happening in the video?"
ANSWER="Yes"

# Number of available GPUs (0-7 means 8 GPUs)
NUM_GPUS=8

# Loop through models and assign each to a GPU in round-robin fashion
for i in "${!MODELS[@]}"; do
    GPU=$((i % NUM_GPUS))  # Cycle through GPUs 0-7
    MODEL_NAME="${MODELS[$i]}"
    
    echo "Running model: $MODEL_NAME on GPU: $GPU"

    # Run the command and send to background
    CUDA_VISIBLE_DEVICES=$GPU python cam_motion_ci.py \
        --score_model "$MODEL_NAME" \
        --batch_size "$BATCH_SIZE" \
        --question "$QUESTION" \
        --answer "$ANSWER" &

    sleep 2  # Small delay to avoid overwhelming the scheduler
done

# Wait for all background jobs to finish
wait

echo "All models finished execution."
