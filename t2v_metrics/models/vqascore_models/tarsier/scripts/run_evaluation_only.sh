#!/bin/bash
export AZURE_ENDPOINT=...
export OPENAI_API_KEY=...

pred_file=$1
benchmarks=${@:2}
benchmarks=${benchmarks:-"all"}

python3 -m evaluation.evaluate \
    --pred_file $pred_file \
    --benchmarks $benchmarks
