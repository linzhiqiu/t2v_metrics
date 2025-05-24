export DETECTRON2_DATASETS="/path/to/detectron2_data"
export PYTHONPATH="$HOME/occhi/apps/detection:$PYTHONPATH"

python3 tools/lazyconfig_train_net_pe.py \
--num-gpus 8 \
--eval-only \
"$@"