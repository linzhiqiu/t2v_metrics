#!/bin/bash

model_path=$1
n_frames=${2:-8}
max_new_tokens=${3:-512}
top_p=${4:-0.8}
temperature=${5:-0}

python3 -m tasks.demo_cli \
    --model_name_or_path $model_path \
    --config "configs/tarser2_default_config.yaml" \
    --max_n_frames $n_frames \
    --max_new_tokens $max_new_tokens \
    --top_p $top_p \
    --temperature $temperature