# Modified from
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import datasets
import datasets.samplers as samplers

import numpy as np
import torch
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from engine_tta import evaluate_tta
from models import build_model
from torch.utils.data import DataLoader
from util.ema import requires_grad, update_ema


def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--eval_per_epochs", default=1, type=int)
    parser.add_argument("--save_per_epochs", default=1, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--sgd", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", default=0.999, type=float)

    # Variants of Deformable DETR
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--backbone_size",
        default="Gwin384",
        type=str,
        help="backbone size",
    )
    parser.add_argument(
        "--backbone_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--backbone_lrd",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--backbone_layers",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--backbone_init_values",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--backbone_tile_posemb",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--backbone_use_act_checkpoint",
        action="store_true",
        help="If true, we use act_checkpoint in backbone",
    )
    parser.add_argument(
        "--backbone_act_checkpoint_ratio",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--backbone_tta_rope",
        action="store_true",
    )
    parser.add_argument(
        "--backbone_multi_layer",
        action="store_true",
    )

    parser.add_argument(
        "--backbone_win_aug",
        action="store_true",
    )

    parser.add_argument(
        "--backbone_dp",
        default=-1.0,
        type=float,
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=300, type=int, help="Number of query slots"
    )
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument("--use_fed_loss", action="store_true")

    # * Matcher
    parser.add_argument("--assign_first_stage", action="store_true")
    parser.add_argument("--assign_second_stage", action="store_true")
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # dataset parameters
    parser.add_argument("--new_mean_std", action="store_true")
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--bigger", action="store_true")
    parser.add_argument("--lsj", action="store_true")
    parser.add_argument("--lsj_ms", action="store_true")

    parser.add_argument("--lsj_img_size", default=1024, type=int)
    parser.add_argument("--lsj_img_train_min", default=480, type=int)
    parser.add_argument("--lsj_img_size_max", default=-1, type=int)
    parser.add_argument("--lsj_strong_aug", action="store_true")

    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--save_result_dir", default="", type=str)
    parser.add_argument("--test_hflip_aug", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--soft_nms", action="store_true")
    parser.add_argument("--soft_nms_method", default="quad", type=str)
    parser.add_argument("--nms_thresh", default=0.7, type=float)
    parser.add_argument("--quad_scale", default=0.5, type=float)
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")

    parser.add_argument(
        "--resume_norope",
        action="store_true",
        help="resume from checkpoint without rope params",
    )
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--keep_class_embed", action="store_true")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )

    return parser


# lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_vit_lr_decay_rate_vev01(name, lr_decay_rate=1.0, num_layers=12):
    layer_id = num_layers + 1
    if ".positional_embedding" in name or ".conv1" in name or ".ln_pre" in name:
        layer_id = 0
    elif ".resblocks." in name:
        layer_id = int(name[name.find(".resblocks.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def custom_lr(model_without_ddp, args):
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    if "vev01" in args.backbone:
        for p_key, p_value in model_without_ddp.named_parameters():
            if (
                match_name_keywords(p_key, args.lr_backbone_names)
                and p_value.requires_grad
            ):
                p_lr = args.lr_backbone * get_vit_lr_decay_rate_vev01(
                    p_key, args.backbone_lrd, args.backbone_layers
                )
                param_dicts.append(
                    {
                        "params": [p_value],
                        "lr": p_lr,
                    }
                )
                print(f"param_name: {p_key}, lr: {p_lr}")
    else:
        param_groups_backbone = {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        }
        param_dicts.append(param_groups_backbone)

    return param_dicts


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model:", model_without_ddp)
    for n, p in model_without_ddp.named_parameters():
        print(n)
    print("number of params:", n_parameters)

    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        print(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            if args.dataset_file == "lvis":
                sampler_train = samplers.RepeatFactorTrainingSampler(dataset_train)
            else:
                sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    if args.lsj_ms:
        collator = utils.CollatorLSJMultiscale(args.lsj_img_size, args.tta)
    elif args.lsj:
        lsj_img_size_colla = (
            args.lsj_img_size_max if args.lsj_img_size_max > 0 else args.lsj_img_size
        )
        collator = utils.CollatorLSJ(lsj_img_size_colla, args.tta)
    else:
        collator = utils.collate_fn

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    param_dicts = custom_lr(model_without_ddp, args)

    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])
    
    if args.tta:
        evaluate_fn = evaluate_tta
    else:
        evaluate_fn = evaluate

    output_dir = Path(args.output_dir)
    if args.auto_resume:
        resumed_ckpt = os.path.join(args.output_dir, "checkpoint.pth")
        if os.path.exists(resumed_ckpt):
            args.resume = resumed_ckpt
            args.finetune = None

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        state_dict = checkpoint["model"]
        for k in list(state_dict.keys()):
            if "class_embed" in k and not args.keep_class_embed:
                print("removing", k)
                del state_dict[k]
            if "freqs" in k:
                print("removing", k)
                del state_dict[k]

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            state_dict, strict=False
        )
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not (k.endswith("total_params") or k.endswith("total_ops"))
        ]
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))

        if "epoch" in checkpoint:
            print("finetuning from epoch", checkpoint["epoch"])

        if args.ema:
            ema.load_state_dict(
                checkpoint["ema"] if "ema" in checkpoint else state_dict, strict=False
            )

    if args.resume:
        print("Resuming training from {}".format(args.resume))
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        if args.resume_norope:
            state_dict = checkpoint["model"]
            for k in list(state_dict.keys()):
                if "freqs" in k:
                    print("removing", k)
                    del state_dict[k]

            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
                state_dict, strict=False
            )
            if args.ema:
                ema.load_state_dict(
                    checkpoint["ema"] if "ema" in checkpoint else state_dict,
                    strict=False,
                )
        else:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
                checkpoint["model"], strict=False
            )
            if args.ema:
                ema.load_state_dict(
                    checkpoint["ema"] if "ema" in checkpoint else state_dict,
                    strict=False,
                )
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not (k.endswith("total_params") or k.endswith("total_ops"))
        ]
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            import copy

            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    "Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler."
                )
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(
                    map(lambda group: group["initial_lr"], optimizer.param_groups)
                )
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint["epoch"] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate_fn(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                args.test_hflip_aug,
                args.tta,
                args.soft_nms,
                ema if args.ema else None,
                args.save_result,
                args.save_result_dir,
                soft_nms_method=args.soft_nms_method,
                nms_thresh=args.nms_thresh,
                quad_scale=args.quad_scale,
                lsj_img_size=args.lsj_img_size,
            )
        torch.cuda.empty_cache()

    if args.eval:
        test_stats, coco_evaluator = evaluate_fn(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
            args.test_hflip_aug,
            args.tta,
            args.soft_nms,
            ema if args.ema else None,
            args.save_result,
            args.save_result_dir,
            soft_nms_method=args.soft_nms_method,
            nms_thresh=args.nms_thresh,
            quad_scale=args.quad_scale,
            lsj_img_size=args.lsj_img_size,
        )

        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )
        return

    print("Start training")
    start_time = time.time()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            ema if args.ema else None,
            ema_decay=args.ema_decay,
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 5 epochs
            if (
                (epoch + 1) % args.lr_drop == 0
                or (epoch + 1) % args.save_per_epochs == 0
                or epoch + 1 == args.epochs
            ):
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                ckpt_dict = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if args.ema:
                    ckpt_dict["ema"] = ema.state_dict()
                utils.save_on_master(
                    ckpt_dict,
                    checkpoint_path,
                )

        torch.cuda.empty_cache()
        if epoch % args.eval_per_epochs == 0 or epoch + 1 == args.epochs:
            test_stats, coco_evaluator = evaluate_fn(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                args.test_hflip_aug,
                args.tta,
                args.soft_nms,
                ema if args.ema else None,
                args.save_result,
                args.save_result_dir,
                soft_nms_method=args.soft_nms_method,
                nms_thresh=args.nms_thresh,
                quad_scale=args.quad_scale,
                lsj_img_size=args.lsj_img_size,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                output_dir / "eval" / name,
                            )
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
