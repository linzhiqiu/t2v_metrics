#!/bin/bash

model_path=$1
max_n_frames=${2:-8}

export MODEL_PATH=$model_path
export MAX_N_FRAMES=$max_n_frames

python3 -m tasks.demo_gradio