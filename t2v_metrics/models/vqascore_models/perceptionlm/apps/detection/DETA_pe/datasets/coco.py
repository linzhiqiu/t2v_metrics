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
import random
from pathlib import Path

import datasets.transforms as T
import torch
import torch.utils.data
import torchvision.transforms.functional as F
from pycocotools import mask as coco_mask
from util.misc import get_local_rank, get_local_size

from .torchvision_datasets import CocoDetection as TvCocoDetection



class CocoDetection(TvCocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        cache_mode=False,
        local_rank=0,
        local_size=1,
        test_hflip_aug=False,
        tta=False,
        is_train=False,
        lsj_img_size=1824,
    ):
        super(CocoDetection, self).__init__(
            img_folder,
            ann_file,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
        )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.test_hflip_aug = test_hflip_aug
        self.tta = tta
        if lsj_img_size == 1728: # for back-compatibility
            self.tta_image_size = [1536, 1152,]
        else:
            self.tta_image_size = [1728, 1536, 1344,]
        
        self.is_train = is_train

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if self.test_hflip_aug:
            flipped_img = torch.flip(img, dims=[-1])
            new_img = torch.cat([img, flipped_img], dim=0)
            return new_img, target

        elif self.tta:
            tta_images = [img]
            flipped_img = torch.flip(img, dims=[-1])
            tta_images.append(flipped_img)
            _, height, width = img.shape
            max_size_len = height if height >= width else width
            for new_max_size in self.tta_image_size:
                scale = new_max_size / max_size_len
                new_height, new_width = int(scale * height), int(scale * width)
                new_img = F.resize(img, size=(new_height, new_width))
                tta_images.append(new_img)
                flipped_img = torch.flip(new_img, dims=[-1])
                tta_images.append(flipped_img)
            return tta_images, target
        else:
            return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, bigger):

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if "train" in image_set:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if "val" in image_set or "test" in image_set:
        scales = [800]

    max_size = 1333
    if bigger:
        scales = [int(1.5 * s) for s in scales]
        max_size = 2000

    if image_set == "train":
        augmentation_list = [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose(
                    [
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=max_size),
                    ]
                ),
            ),
            normalize,
        ]

        return T.Compose(augmentation_list)

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize(scales, max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def make_coco_transforms_lsj(
    image_set, image_size, lsj_img_train_min=480, lsj_strong_aug=False
):
    """
    Reference: https://github.com/facebookresearch/detectron2/blob/main/projects/ViTDet/configs/common/coco_loader_lsj.py

    import detectron2.data.transforms as T
    from detectron2 import model_zoo
    from detectron2.config import LazyCall as L

    # Data using LSJ
    image_size = 1024
    dataloader = model_zoo.get_config("common/data/coco.py").dataloader
    dataloader.train.mapper.augmentations = [
        L(T.RandomFlip)(horizontal=True),  # flip first
        L(T.ResizeScale)(
            min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
        ),
        L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
    ]
    dataloader.train.mapper.image_format = "RGB"
    dataloader.train.total_batch_size = 64
    # recompute boxes due to cropping
    dataloader.train.mapper.recompute_boxes = True

    dataloader.test.mapper.augmentations = [
        L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
    ]
    """

    """
    In our implementation, we simulate lsj data augmentation by:
    (1) first the following augmentations 
    (2) then padding to (image_size, image_size) in collator, see util/misc/collate_fn_lsj.py
    """
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if "train" in image_set:
        scales = [scale for scale in range(lsj_img_train_min, image_size, 32)]
    if "val" in image_set or "test" in image_set or "unlabel" in image_set:
        scales = [image_size - 32]

    # max_size = 1333
    # if bigger:
    #     scales = [int(1.5 * s) for s in scales]
    #     max_size = 2000
    max_size = image_size - 32  # for some wired bugs

    augmentation_list = []
    if "train" in image_set:
        if lsj_strong_aug:
            augmentation_list.extend(
                [
                    T.ColorJitter((0.4, 0.4, 0.4, 0.1), p=0.5),
                    T.RandomGrayscale(p=0.2),
                    # T.RandomErasingP05(),
                ]
            )
        augmentation_list.extend(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    # similar to (T.ResizeScale)(min_scale=0.1, max_scale=1.0, target_height=image_size, target_width=image_size) and pad
                    T.RandomResize(scales, max_size=max_size),
                    # similar to (T.ResizeScale)(min_scale=1.0, max_scale=2.0, target_height=image_size, target_width=image_size) and crop
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize([max_size], max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )
        return T.Compose(augmentation_list)

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize(scales, max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    if args.lsj:
        coco_transform = make_coco_transforms_lsj(
            image_set,
            args.lsj_img_size,
            args.lsj_img_train_min,
            args.lsj_strong_aug,
        )
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
        test_hflip_aug=args.test_hflip_aug,
        tta=args.tta,
        is_train=("train" in image_set),
        lsj_img_size=args.lsj_img_size,
    )
    return dataset
