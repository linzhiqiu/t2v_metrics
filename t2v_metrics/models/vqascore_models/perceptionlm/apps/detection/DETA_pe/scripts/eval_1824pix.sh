
EXP_DIR="/checkpoint/vision_encoder/d2_output/coco_sota/eval"


python -m torch.distributed.launch --nproc_per_node=8 \
--master_port=12345 --use_env main.py \
--output_dir ${EXP_DIR} \
--with_box_refine --two_stage \
--num_feature_levels 5 --num_queries 900 \
--dim_feedforward 2048 --dropout 0.0 --cls_loss_coef 1.0 \
--assign_first_stage --assign_second_stage \
--epochs 24 --lr_drop 20 \
--lr 5e-5 --lr_backbone 5e-5 --batch_size 1 \
--backbone pev1 \
--backbone_size Gwin384 \
--backbone_init_values 0.1 \
--backbone_tile_posemb True \
--backbone_lrd 0.9 --backbone_layers 50 \
--num_workers 4 \
--coco_path /checkpoint/vision_encoder/public_data/coco \
--lsj --lsj_img_size 1824 \
--backbone_use_act_checkpoint --backbone_act_checkpoint_ratio 1.0 \
--eval \
--resume /checkpoint/vision_encoder/d2_output/coco_sota/finetune_further_spatial_Gwin384_cocoep3_1824pix_8node/checkpoint.pth \
"$@"
