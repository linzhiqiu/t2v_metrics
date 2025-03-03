# Evaluate on camera-centric benchmark
# CUDA_VISIBLE_DEVICES=2 python cam_motion_temp.py --score_model umt-b16-25m-clip
# CUDA_VISIBLE_DEVICES=4 python cam_motion_temp.py --score_model umt-b16-25m-itm
# CUDA_VISIBLE_DEVICES=1 python cam_motion_temp.py --score_model languagebind-video-v1.5-ft
# CUDA_VISIBLE_DEVICES=2 python cam_motion_temp.py --score_model languagebind-video-ft
# CUDA_VISIBLE_DEVICES=3 python cam_motion_temp.py --score_model internvideo2-1b-stage2-clip
# CUDA_VISIBLE_DEVICES=4 python cam_motion_temp.py --score_model internvideo2-1b-stage2-itm
# CUDA_VISIBLE_DEVICES=5 python cam_motion_temp.py --score_model umt-l16-25m-clip
# CUDA_VISIBLE_DEVICES=6 python cam_motion_temp.py --score_model umt-l16-25m-itm
import argparse
import os
import sys
import t2v_metrics
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

ROOT = Path("/data3/zhiqiul/video_annotation")
VIDEO_ROOT = Path("/data3/zhiqiul/video_annotation/videos")
# Get the absolute path of the video_annotation/ folder
# Add it to sys.path
sys.path.append(os.path.abspath(ROOT))

from benchmark import labels_as_dict, BinaryTask

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--score_model",
    type=str,
    # default="umt-b16-25m-clip",
    # default="umt-l16-25m-clip",
    # default="umt-l16-25m-itm",
    # default="umt-b16-25m-itm",
    # default="internvideo2-1b-stage2-clip",
    # default="internvideo2-1b-stage2-itm",
    default="languagebind-video-v1.5-ft",
    # default="languagebind-video-ft",
    # default="languagebind-video-v1.5",
    # default="languagebind-video",
    # default="languagebind-video-v1.5-huge-ft",
    help="The score model to use",
)
parser.add_argument(
    "--root",
    type=str,
    help="The root directory for saved datasets",
    default=ROOT,
)
parser.add_argument(
    "--video_root",
    type=str,
    help="The root directory for saved videos",
    default=VIDEO_ROOT,
)
parser.add_argument(
    "--video_label_file",
    type=str,
    help="The video label to use",
    # default="video_labels/cam_motion-20250219_0338/label_names.json", @ noisy 4000
    # default="video_labels/cam_motion-cam_setup-20250218_1042/label_names.json", # good initial set of around 2000
    # default="video_labels/cam_motion-20250223_0313/label_names_subset.json", # only a subset of 300 videos
    # default="video_labels/cam_motion-20250223_2308/label_names_selected.json", # the full set of 2000 videos for cam-centric
    # default="video_labels/cam_motion-cam_setup-20250224_0130/label_names_selected.json", # the full set of 2384 videos for ground-centric + shotcomp
    default="video_labels/cam_motion-20250227_0326ground_and_camera/label_names_selected.json", # Finalized cam-centric benchmark
    nargs="?",
)
parser.add_argument(
    "--save_dir",
    type=str,
    help="The directory to save the results",
    default="./temp_better_question/",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="The batch size to use"
)
# parser.add_argument(
#     "--question",
#     default=None,
#     type=str
# )
# parser.add_argument(
#     "--answer",
#     default=None,
#     type=str
# )
# parser.add_argument(
#     "--mode",
#     default="vqa",
#     type=str,
#     choices=["vqa", "retrieval"]
# )
args = parser.parse_args()


print(f"Using score model: {args.score_model}")

score_model = t2v_metrics.get_score_model(model=args.score_model)

# if type of score_model is VQAScoreModel
if isinstance(score_model, t2v_metrics.VQAScore):
    mode = "vqa"
    score_kwargs = {
        "question_template": "{} Please only answer Yes or No.",
        "answer_template": "Yes"
    }
else:
    mode = "retrieval"
    score_kwargs = {}

video_label_str = "_".join(args.video_label_file.split("/")[-2:])[:-5]

SAVE_DIR = Path(args.save_dir) / f"{video_label_str}" 
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

all_labels = labels_as_dict(root=args.root, video_label_file=args.video_label_file, video_root=args.video_root)

print(f"Loaded {len(all_labels)} labels")
label_counts = {}
for label_name in all_labels:
    pos = len(all_labels[label_name]['pos'])
    neg = len(all_labels[label_name]['neg'])
    label_counts[label_name] = pos + neg

for label_name in all_labels:
    print(
        f"{all_labels[label_name]['label_name']:100s}: Positives: {len(all_labels[label_name]['pos']):10d}, Negatives: {len(all_labels[label_name]['neg']):10d}"
    )

all_labels_scores = {}
for label_name in all_labels:
    LABEL_SAVE_DIR = SAVE_DIR / label_name
    if not LABEL_SAVE_DIR.exists():
        LABEL_SAVE_DIR.mkdir(parents=True)
    LABEL_SAVE_PATH = LABEL_SAVE_DIR / f"{args.score_model}_scores.pt"

    dataset = BinaryTask(label_dict=all_labels[label_name])

    if LABEL_SAVE_PATH.exists():
        print(f"Skipping {label_name}")
        all_labels_scores[label_name] = torch.load(LABEL_SAVE_PATH, weights_only=False)
    else:
        scores = score_model.batch_forward(
            dataset,
            batch_size=args.batch_size,
            **score_kwargs
        )
        scores = scores[:, 0, 0].cpu().numpy() # (num_videos, 1, 1) -> (num_videos,)

        results = dataset.evaluate_scores(
            scores,
            plot_path=LABEL_SAVE_DIR / f"{args.score_model}_pr_curve.jpeg",
        )
        all_labels_scores[label_name] = {
            "scores": scores,
            "results": results,
        }
        torch.save(all_labels_scores[label_name], LABEL_SAVE_PATH)

# Sort by best AP
# sorted_labels = sorted(all_labels_scores, key=lambda x: all_labels_scores[x]["results"]["ap"], reverse=True)
# print(f"Using Score Model: {args.score_model}")
header = f"{args.score_model:50s} {'AP':>10} {'ROC AUC':>10} {'F1':>10} {'Threshold':>12}"
separator = "-" * len(header)

print(header)
print(separator)

for label_name, scores in all_labels_scores.items():
    ap = scores["results"]["ap"]
    roc_auc = scores["results"]["roc_auc"]
    optimal_f1 = scores["results"]["optimal_f1"]
    optimal_threshold = scores["results"]["optimal_threshold"]

    print(
        f"{label_name.rsplit('.')[-1]:50s} {ap:>10.4f} {roc_auc:>10.4f} {optimal_f1:>10.4f} {optimal_threshold:>12.4f}"
    )
