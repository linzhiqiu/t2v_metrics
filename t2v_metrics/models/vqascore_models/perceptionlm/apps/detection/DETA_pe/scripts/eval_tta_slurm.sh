#!/bin/bash

#SBATCH --qos=vision_encoder_high
#SBATCH --account=vision_encoder
#SBATCH --job-name=det
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=/checkpoint/vision_encoder/d2_output/slurm_logs/coco_sota/eval_tta_slurm/%j.out
#SBATCH --error=/checkpoint/vision_encoder/d2_output/slurm_logs/coco_sota/eval_tta_slurm/%j.err
#SBATCH --time=23:00:00

module load cuda/12.1
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

read -ra my_array <<< $head_node_ip
export LOGLEVEL=INFO

echo head_node_ip $head_node_ip
echo endpoint "${head_node_ip}:29500"

EXP_DIR="/checkpoint/vision_encoder/d2_output/coco_sota/eval_tta_slurm"


# srun \
# torchrun \
srun \
python -m torch.distributed.run \
--nnodes 8 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_endpoint "${my_array[0]}:29500" \
--rdzv_backend c10d \
main.py \
--output_dir ${EXP_DIR} \
--with_box_refine --two_stage \
--num_feature_levels 5 --num_queries 2000 \
--dim_feedforward 2048 --dropout 0.0 --cls_loss_coef 1.0 \
--assign_first_stage --assign_second_stage \
--epochs 12 --lr_drop 10 \
--lr 5e-5 --lr_backbone 5e-5 --batch_size 1 \
--backbone pev1 \
--backbone_size Gwin384 \
--backbone_init_values 0.1 \
--backbone_tile_posemb True \
--backbone_lrd 0.9 --backbone_layers 50 \
--num_workers 4 \
--coco_path /checkpoint/vision_encoder/public_data/coco \
--lsj --lsj_img_size 1728 \
--backbone_use_act_checkpoint --backbone_act_checkpoint_ratio 1.0 \
--eval \
--resume /checkpoint/vision_encoder/d2_output/coco_sota/finetune_spatial_Gwin384_cocoep12_1728pix_8node/checkpoint.pth \
--soft_nms \
--tta \
"$@"
