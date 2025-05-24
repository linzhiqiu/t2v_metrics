from functools import partial

import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.modeling import SimpleFeaturePyramid, ViT
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.solver import WarmupParamScheduler
from detectron2_pe.modeling import PEv1_det, get_vit_lr_decay_rate_pev1
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ..COCO.mask_rcnn_vitdet_b_100ep import (  # dataloader,; model,; get_vit_lr_decay_rate,
    lr_multiplier, optimizer, train)
from ..common.coco_loader_lsj import dataloader

train.init_checkpoint = "/checkpoint/vision_encoder/pev1/pev1_rc2_d2.pt"
train.output_dir = (
    "/checkpoint/vision_encoder/d2_output/coco/mask_rcnn_PEcore_G_coco36ep"
)

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

model.pixel_mean = [127, 127, 127]
model.pixel_std = [127, 127, 127]
model.input_format = "RGB"


img_size = 1024
embed_dim, depth, num_heads, mlp_ratio, dp = 1536, 50, 16, 8960 / 1536, 0.5
pretrain_img_size, patch_size, window_size = 512, 16, 32
# 12, 24, 36, 49 for global attention
window_block_indexes = (
    list(range(0, 12)) + list(range(13, 24)) + list(range(25, 36)) + list(range(37, 49))
)
# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(PEv1_det)(  # Single-scale ViT backbone
        pretrain_img_size=pretrain_img_size,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=window_size,
        pt_hw_seq_len=32,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=window_block_indexes,
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
        tile_posemb=True,
        use_abs_pos=True,
        pretrain_use_cls_token=False,
        use_act_checkpoint=True,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=img_size,
)

optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate_pev1, lr_decay_rate=0.9, num_layers=50
)

dataloader.train.total_batch_size = 64
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375


lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None
optimizer.lr = 5e-5

train.max_iter = train.max_iter * 3 // 4  # 100ep -> 75ep
lr_multiplier.scheduler.milestones = [
    milestone * 3 // 4 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter
