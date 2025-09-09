# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator, convert_to_xywh
from datasets.data_prefetcher import data_prefetcher
from datasets.panoptic_eval import PanopticEvaluator
from models.utils_softnms import batched_soft_nms
from util.misc import NestedTensor


# Make sure this is consistent with datasets/coco.py 
# TODO: make it configurable
SCALE_RANGES_DICT = {
    1728: [[0, 10000], [32, 10000], [32, 10000],],
    1824: [[0, 10000], [0, 10000], [64, 10000], [64, 10000],],
}


def filter_boxes(boxes, min_scale, max_scale):
    """
    boxes: (N, 4) shape
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return keep


@torch.no_grad()
def evaluate_tta(
    model_no_ema,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    test_hflip_aug,
    tta,
    soft_nms,
    ema=None,
    save_result=False,
    save_result_dir="",
    soft_nms_method="quad",
    nms_thresh=0.7,
    quad_scale=0.5,
    lsj_img_size=1824,
):
    model = model_no_ema if ema is None else ema
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    SCALE_RANGES = SCALE_RANGES_DICT[lsj_img_size]
    IMAGE_SIZE = [lsj_img_size for _ in range(len(SCALE_RANGES))]
    
    prediction_list = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        metric_logger.update(loss=0, class_error=0, loss_bbox=0, loss_ce=0)
        ########################### Begin of inference_one_image ###########################
        if tta:
            assert samples.tensors.shape[0] == 1, "tta only supports batch size 1"
            assert (
                samples.tensors.shape[1] % 3 == 0
            ), "tta requires dimensions of samples.tensors to be divisible by 3"

            all_boxes = []
            all_scores = []
            all_classes = []

            num_scales = samples.tensors.shape[1] // 3
            for scale_ind in range(num_scales):
                first_samples = NestedTensor(
                    samples.tensors[
                        :,
                        scale_ind * 3 : (scale_ind + 1) * 3,
                        : IMAGE_SIZE[scale_ind // 2],
                        : IMAGE_SIZE[scale_ind // 2],
                    ],
                    samples.mask[
                        :,
                        scale_ind,
                        : IMAGE_SIZE[scale_ind // 2],
                        : IMAGE_SIZE[scale_ind // 2],
                    ],
                )

                if scale_ind % 2 == 0:
                    ######## no flip #######
                    outputs = model(first_samples)
                    noaug_results = postprocessors["bbox"](
                        outputs, 
                        orig_target_sizes, 
                        soft_nms=soft_nms,
                        method=soft_nms_method,
                        nms_thresh=nms_thresh,
                        quad_scale=quad_scale,
                    )
                    keep = filter_boxes(
                        noaug_results[0]["boxes"], *SCALE_RANGES[scale_ind // 2]
                    )
                    all_boxes.append(noaug_results[0]["boxes"][keep])
                    all_scores.append(noaug_results[0]["scores"][keep])
                    all_classes.append(noaug_results[0]["labels"][keep])
                else:
                    ######## flipped #######
                    flipped_outputs = model(first_samples)
                    flipped_pred_logits = flipped_outputs["pred_logits"]
                    flipped_pred_boxes = flipped_outputs["pred_boxes"]
                    reflipped_pred_boxes = flipped_pred_boxes[
                        :, :, [0, 1, 2, 3]
                    ] * torch.as_tensor([-1, 1, 1, 1]).to(
                        flipped_pred_boxes.device
                    ) + torch.as_tensor(
                        [1, 0, 0, 0]
                    ).to(
                        flipped_pred_boxes.device
                    )
                    new_outputs = {}
                    new_outputs["pred_logits"] = flipped_pred_logits
                    new_outputs["pred_boxes"] = reflipped_pred_boxes
                    new_results = postprocessors["bbox"](
                        new_outputs, 
                        orig_target_sizes, 
                        soft_nms=soft_nms,
                        method=soft_nms_method,
                        nms_thresh=nms_thresh,
                        quad_scale=quad_scale,
                    )
                    keep = filter_boxes(
                        new_results[0]["boxes"], *SCALE_RANGES[scale_ind // 2]
                    )
                    all_boxes.append(new_results[0]["boxes"][keep])
                    all_scores.append(new_results[0]["scores"][keep])
                    all_classes.append(new_results[0]["labels"][keep])

            ######## merge #######
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_classes = torch.cat(all_classes, dim=0)

            keep_inds, updated_scores = batched_soft_nms(
                all_boxes,
                all_scores,
                all_classes,
                method=soft_nms_method,
                threshold=nms_thresh,
                quad_scale=quad_scale,
            )
            merged_scores = updated_scores
            merged_classes = all_classes[keep_inds]
            merged_boxes = all_boxes[keep_inds]

            results = [
                {
                    "boxes": merged_boxes,
                    "scores": merged_scores,
                    "labels": merged_classes,
                }
            ]
        else:
            outputs = model(samples)
            results = postprocessors["bbox"](outputs, orig_target_sizes)

        ########################### End of inference_one_image ###########################
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        for target, output in zip(targets, results):
            res_cpu = {
                target["image_id"].item(): {
                    "boxes": output["boxes"].cpu(),
                    "labels": output["labels"].cpu(),
                    "scores": output["scores"].cpu(),
                }
            }
            prediction_list.append(res_cpu)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if save_result:
        from torch import distributed as dist

        os.makedirs(save_result_dir, exist_ok=True)

        rank = dist.get_rank()
        torch.save(
            prediction_list,
            os.path.join(save_result_dir, f"val2017_prediction_{rank}.pth"),
        )

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    return stats, coco_evaluator
