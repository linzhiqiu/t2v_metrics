#!/bin/bash

#SBATCH --qos=vision_encoder
#SBATCH --account=vision_encoder
#SBATCH --job-name=det
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=/checkpoint/vision_encoder/d2_output/slurm_logs/coco_sota/pretrain_spatial_Gwin384_o365ep12_1024pix_16node/%j.out
#SBATCH --error=/checkpoint/vision_encoder/d2_output/slurm_logs/coco_sota/pretrain_spatial_Gwin384_o365ep12_1024pix_16node/%j.err
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


EXP_DIR="/checkpoint/vision_encoder/d2_output/coco_sota/pretrain_spatial_Gwin384_o365ep12_1024pix_16node"

srun \
torchrun \
--nnodes 16 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_endpoint "${my_array[0]}:29500" \
--rdzv_backend c10d \
main.py \
--output_dir ${EXP_DIR} \
--with_box_refine --two_stage \
--num_feature_levels 5 --num_queries 900 \
--dim_feedforward 2048 --dropout 0.0 --cls_loss_coef 1.0 \
--assign_first_stage --assign_second_stage \
--epochs 12 --lr_drop 10 \
--lr_backbone 2e-4 \
--backbone pev1 \
--backbone_size Gwin384 \
--backbone_path /checkpoint/vision_encoder/pev1/pe_spatial_G14_448_16patch384pix.pth \
--backbone_init_values 0.1 \
--backbone_tile_posemb True \
--backbone_lrd 0.9 --backbone_layers 50 \
--dataset_file objects365 \
--coco_path /checkpoint/vision_encoder/public_data/objects365_v2 \
--lsj --lsj_img_size 1024 \
--backbone_use_act_checkpoint --backbone_act_checkpoint_ratio 1.0 \
--eval_per_epochs 2 \
--save_per_epochs 1 \
--auto_resume \
"$@"
