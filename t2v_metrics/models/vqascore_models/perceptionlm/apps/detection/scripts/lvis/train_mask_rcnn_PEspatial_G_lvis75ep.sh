#!/bin/bash

#SBATCH --qos=vision_encoder
#SBATCH --account=vision_encoder
#SBATCH --job-name=det
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=/checkpoint/vision_encoder/d2_output/slurm_logs/lvis/train_mask_rcnn_PEspatial_G_lvis75ep/%j.out
#SBATCH --error=/checkpoint/vision_encoder/d2_output/slurm_logs/lvis/train_mask_rcnn_PEspatial_G_lvis75ep/%j.err
#SBATCH --time=96:00:00

module load cuda/12.1
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

read -ra my_array <<< $head_node_ip
export LOGLEVEL=INFO

echo head_node_ip $head_node_ip
echo endpoint "${head_node_ip}:29500"

export DETECTRON2_DATASETS="/path/to/detectron2_data"
export PYTHONPATH="$HOME/occhi/apps/detection:$PYTHONPATH"

srun \
torchrun \
--nnodes 8 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_endpoint "${my_array[0]}:29500" \
--rdzv_backend c10d \
tools/lazyconfig_train_net_pe_slurm.py \
--resume \
--config-file projects/ViTDet/configs/LVIS/mask_rcnn_PEspatial_G_lvis75ep.py \
optimizer.lr=5e-5 \
train.init_checkpoint="/checkpoint/vision_encoder/pev1/pe_spatial_G14_16patch.pth" \
train.output_dir="/checkpoint/vision_encoder/d2_output/lvis/train_mask_rcnn_PEspatial_G_lvis75ep" \
model.backbone.net.init_values=0.1 \
model.backbone.net.use_act_checkpoint=True \
"$@"
