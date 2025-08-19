#!/bin/bash

# Available GPUs
gpus=(1 2 3 5 6)

# Models to test
models=(
    # "qwen2.5-vl-bal-cap-fps2"
    # "qwen2.5-vl-bal-cap-fps4"
    'qwen2.5-vl-imb-cap-fps2'
    "qwen2.5-vl-bal-cap-fps8"
    "qwen2.5-vl-bal-imb-cap-fps2"
    "qwen2.5-vl-bal-imb-cap-fps4"
    "qwen2.5-vl-bal-imb-cap-fps8"
)

# Distribute models across GPUs with staggering
for i in "${!models[@]}"; do
    gpu_index=$((i % ${#gpus[@]}))
    gpu=${gpus[$gpu_index]}
    model=${models[$i]}
    
    # Stagger the start of each GPU's workload
    sleep $((i * 2))
    
    # Run both tests SEQUENTIALLY for this model on the assigned GPU
    (
        echo "GPU $gpu: Starting tests for model $model"
        
        # First test
        # echo "GPU $gpu: Running better_question test for $model"
        # CUDA_VISIBLE_DEVICES=$gpu python cam_motion_better_question_test_temp.py --use_testset --score_model "$model"
        
        # # Second test - only starts after first test completes
        # echo "GPU $gpu: Running pairwise test for $model"
        # CUDA_VISIBLE_DEVICES=$gpu python cam_motion_pairwise_test_temp.py --score_model "$model"

        # Third test - only starts after first test completes
        echo "GPU $gpu: Running pairwise test complex caption for $model"
        CUDA_VISIBLE_DEVICES=$gpu python cam_motion_pairwise_test_complex_caption_temp.py --score_model "$model"
        
        echo "GPU $gpu: Completed all tests for $model"
    ) &
done

# Wait for all background processes to complete
wait
echo "All tests completed."