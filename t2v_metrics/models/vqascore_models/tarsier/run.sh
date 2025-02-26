MODEL_NAME_OR_PATH="omni-research/Tarsier2-Recap-7b"
VIDEO_FILE="assets/videos/coffee.gif" # Or try your own example, could be images (include gif images), videos.

python3 -m tasks.inference_quick_start \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --config configs/tarser2_default_config.yaml \
  --instruction "Describe the video in detail." \
  --input_path $VIDEO_FILE