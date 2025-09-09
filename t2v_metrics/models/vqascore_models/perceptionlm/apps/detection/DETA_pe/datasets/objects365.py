# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import datasets.transforms as T

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from util.misc import get_local_rank, get_local_size

from .coco import CocoDetection, make_coco_transforms, make_coco_transforms_lsj
from .torchvision_datasets import CocoDetection as TvCocoDetection


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided Objects365 path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (
            root / "train",
            root / "annotations" / "zhiyuan_objv2_train_fixmiss.json",
        ),
        "val": (root / "val", root / "annotations" / "zhiyuan_objv2_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    if args.lsj:
        coco_transform = make_coco_transforms_lsj(image_set, args.lsj_img_size)
    else:
        coco_transform = make_coco_transforms(image_set, args.bigger)
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=coco_transform,
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
    )
    return dataset
