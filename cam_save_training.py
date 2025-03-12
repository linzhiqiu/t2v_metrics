# Generate a camera-centric training set
# CUDA_VISIBLE_DEVICES=2 python cam_save_training.py
import argparse
import os
import sys
import t2v_metrics
import torch
from torch.utils.data import Dataset
from pathlib import Path

ROOT = Path("/data3/zhiqiul/video_annotation")
VIDEO_ROOT = Path("/data3/zhiqiul/video_annotation/videos")
VIDEO_LABEL_DIR = ROOT / "video_labels/"
# Get the absolute path of the video_annotation/ folder
# Add it to sys.path
sys.path.append(os.path.abspath(ROOT))

from benchmark import labels_as_dict, BinaryTask
from pairwise_benchmark import (
    generate_pairwise_datasets,
)

import argparse

parser = argparse.ArgumentParser()
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
    default="video_labels/cam_motion-20250227_0326ground_and_camera/label_names_selected.json", # Finalized cam-centric benchmark
    nargs="?",
)
parser.add_argument(
    "--video_label_file_scene_movement",
    type=str,
    help="The video label to use for scene movement",
    default="video_labels/cam_motion-20250227_0326ground_and_camera/scene_movement.json",  # Finalized cam-centric benchmark
    nargs="?",
)
parser.add_argument(
    "--video_label_dir",
    type=str,
    help="The video label folder to use",
    default=VIDEO_LABEL_DIR,
)
parser.add_argument(
    "--sampling",
    type=str,
    help="The sampling method to use (random meaning random sampling, top meaning taking the first N samples)",
    default="top",
    choices=["random", "top"],
)
parser.add_argument(
    "--max_samples",
    type=int,
    help="The maximum number of samples to use",
    default=80,
)
parser.add_argument(
    "--seed",
    type=int,
    help="The seed to use",
    default=0,
)
parser.add_argument(
    "--train_ratio", type=float, help="The ratio of training samples", default=0.5
)
parser.add_argument(
    "--save_file",
    type=str,
    help="The directory to save the results",
    default="./camera_centric_trainset.json",
)
args = parser.parse_args()


mode = "vqa"
score_kwargs = {
    "question_template": "{} Please only answer Yes or No.",
    "answer_template": "Yes"
}

sampling_str = "top" if args.sampling == "top" else f"random_seed_{args.seed}"
folder_name = f"test_ratio_{1 - args.train_ratio:.2f}_num_{args.max_samples}_sampling_{sampling_str}"
video_label_str = "_".join(args.video_label_file.split("/")[-2:])[:-5]

all_labels = labels_as_dict(root=args.root, video_label_file=args.video_label_file, video_root=args.video_root)
scene_movement_labels = labels_as_dict(root=args.root, video_label_file=args.video_label_file_scene_movement, video_root=args.video_root)

label_to_name = {
    "cam_motion.steadiness_and_movement.fixed_camera": "Static",
    "cam_motion.camera_centric_movement.zoom_in.has_zoom_in": "Zoom In",
    "cam_motion.camera_centric_movement.zoom_out.has_zoom_out": "Zoom Out",
    "cam_motion.camera_centric_movement.forward.has_forward_wrt_camera": "Move In",
    "cam_motion.camera_centric_movement.backward.has_backward_wrt_camera": "Move Out",
    "cam_motion.camera_centric_movement.upward.has_upward_wrt_camera": "Move Up",
    "cam_motion.camera_centric_movement.downward.has_downward_wrt_camera": "Move Down",
    "cam_motion.camera_centric_movement.rightward.has_rightward": "Move Right",
    "cam_motion.camera_centric_movement.leftward.has_leftward": "Move Left",
    "cam_motion.camera_centric_movement.pan_right.has_pan_right": "Pan Right",
    "cam_motion.camera_centric_movement.pan_left.has_pan_left": "Pan Left",
    "cam_motion.camera_centric_movement.tilt_up.has_tilt_up": "Tilt Up",
    "cam_motion.camera_centric_movement.tilt_down.has_tilt_down": "Tilt Down",
    "cam_motion.camera_centric_movement.roll_clockwise.has_roll_clockwise": "Roll Clockwise",
    "cam_motion.camera_centric_movement.roll_counterclockwise.has_roll_counterclockwise": "Roll Counterclockwise",
    "cam_motion.scene_movement.dynamic_scene": "Dynamic Scene",
    "cam_motion.scene_movement.mostly_static_scene": "Mostly Static Scene",
    "cam_motion.scene_movement.static_scene": "Static Scene",
}

name_to_label = {v: k for k, v in label_to_name.items()}

datasets = generate_pairwise_datasets(
    max_samples=args.max_samples,
    sampling=args.sampling,
    seed=args.seed,
    root=args.root,
    video_root=args.video_root,
    video_labels_dir=args.video_label_dir,
    train_ratio=args.train_ratio,
)
test_videos = datasets["sampled_tasks"]["test_videos"]
train_videos = datasets["sampled_tasks"]["train_videos"]
print(f"Using {len(train_videos)} training videos and {len(test_videos)} testing videos")

print(f"Loaded {len(all_labels)} labels and {len(scene_movement_labels)} scene movement labels")
label_counts = {}
test_labels = {}
train_labels = {}
scene_labels = {
    "Dynamic Scene": {},
    "Mostly Static Scene": {},
    "Static Scene": {},
}
scene_labels_balanced = {
    "Dynamic Scene": {},
    "Mostly Static Scene": {},
    "Static Scene": {},
}
for label_name in all_labels:
    pos = all_labels[label_name]['pos']
    neg = all_labels[label_name]['neg']
    label_counts[label_name] = {
        "pos": pos,
        "neg": neg,
        "total": pos + neg
    }
    pos_test = [v for v in all_labels[label_name]['pos'] if v in test_videos]
    neg_test = [v for v in all_labels[label_name]['neg'] if v in test_videos]
    test_labels[label_name] = {
        "pos": pos_test,
        "neg": neg_test,
        "total": pos_test + neg_test
    }
    pos_train = [v for v in all_labels[label_name]['pos'] if v in train_videos]
    neg_train = [v for v in all_labels[label_name]['neg'] if v in train_videos]
    train_labels[label_name] = {
        "pos": pos_train,
        "neg": neg_train,
        "total": pos_train + neg_train
    }
    for scene_name in scene_labels:
        scene_label = name_to_label[scene_name]
        scene_info = scene_movement_labels[scene_label]
        pos_scene = [v for v in all_labels[label_name]["pos"] if v in scene_info["pos"]]
        neg_scene = [v for v in all_labels[label_name]["neg"] if v in scene_info["neg"]]
        pos_scene_test = [v for v in pos_scene if v in test_videos]
        neg_scene_test = [v for v in neg_scene if v in test_videos]
        scene_labels[scene_name][label_name] = {
            "pos": pos_scene,
            "neg": neg_scene,
            "pos_test": pos_scene_test,
            "neg_test": neg_scene_test,
            "total": pos_scene + neg_scene
        }

    pos_to_neg_ratio = []
    pos_to_neg_ratio_test = []
    for scene_name in scene_labels_balanced:
        pos_to_neg_ratio.append(len(scene_labels[scene_name][label_name]["pos"]) / len(scene_labels[scene_name][label_name]["neg"]))
        pos_to_neg_ratio_test.append(len(scene_labels[scene_name][label_name]["pos_test"]) / len(scene_labels[scene_name][label_name]["neg_test"]))
    max_ratio = max(pos_to_neg_ratio)
    max_ratio_test = max(pos_to_neg_ratio_test)
    for scene_name in scene_labels_balanced:
        pos_scene = scene_labels[scene_name][label_name]["pos"]
        neg_scene = scene_labels[scene_name][label_name]["neg"]
        if max_ratio >= 1.0:
            # Threshold to have same number of positive and negative samples
            num_samples = min(len(pos_scene), len(neg_scene))
            pos_scene_balanced = pos_scene[:num_samples]
            neg_scene_balanced = neg_scene[:num_samples]
        else:
            # Threshold to have same ratio of positive and negative samples
            num_neg_samples = int(len(pos_scene) / max_ratio)
            pos_scene_balanced = pos_scene
            neg_scene_balanced = neg_scene[:num_neg_samples]

        pos_scene_test = scene_labels[scene_name][label_name]["pos_test"]
        neg_scene_test = scene_labels[scene_name][label_name]["neg_test"]
        if max_ratio_test >= 1.0:
            # Threshold to have same number of positive and negative samples
            num_samples = min(len(pos_scene_test), len(neg_scene_test))
            pos_scene_balanced_test = pos_scene_test[:num_samples]
            neg_scene_balanced_test = neg_scene_test[:num_samples]
        else:
            # Threshold to have same ratio of positive and negative samples
            num_neg_samples = int(len(pos_scene_test) / (max_ratio_test + 1e-6))
            pos_scene_balanced_test = pos_scene_test
            neg_scene_balanced_test = neg_scene_test[:num_neg_samples]
        scene_labels_balanced[scene_name][label_name] = {
            "pos": pos_scene_balanced,
            "neg": neg_scene_balanced,
            "pos_test": pos_scene_balanced_test,
            "neg_test": neg_scene_balanced_test,
            "total": pos_scene_balanced + neg_scene_balanced
        }

for label_name in all_labels:
    pos = label_counts[label_name]['pos']
    neg = label_counts[label_name]['neg']
    pos_test = test_labels[label_name]['pos']
    neg_test = test_labels[label_name]['neg']
    print(
        f"{all_labels[label_name]['label_name']:80s}: Positives (Test/Total): {len(pos_test):5d}/{len(pos):5d}, Negatives (Test/Total): {len(neg_test):5d}/{len(neg):5d}"
    )
    for scene_name in scene_labels:
        scene_label = name_to_label[scene_name]
        scene_info = scene_movement_labels[scene_label]
        pos_scene = scene_labels[scene_name][label_name]["pos"]
        neg_scene = scene_labels[scene_name][label_name]["neg"]
        pos_scene_test = scene_labels[scene_name][label_name]["pos_test"]
        neg_scene_test = scene_labels[scene_name][label_name]["neg_test"]
        pos_scene_balanced = scene_labels_balanced[scene_name][label_name]["pos"]
        neg_scene_balanced = scene_labels_balanced[scene_name][label_name]["neg"]
        print(
            f"{'':25s}{scene_name:25s}: Positives (All/Test/Total): {len(pos_scene):5d}/{len(pos_scene_test):5d}/{len(pos):5d}, Negatives (All/Test/Total): {len(neg_scene):5d}/{len(neg_scene_test):5d}/{len(neg):5d}"
        )
        print(
            f"{'':25s}{scene_name+' Balanced':25s}: Positives (All/Test/Total): {len(pos_scene_balanced):5d}/{len(pos_scene_balanced_test):5d}/{len(pos_scene_balanced):5d}, Negatives (All/Test/Total): {len(neg_scene_balanced):5d}/{len(neg_scene_balanced_test):5d}/{len(neg_scene_balanced):5d}"
        )

import numpy as np
all_labels_scores = {}
all_samples = []
pos_samples = 0
neg_samples = 0
for label_name in all_labels:
    all_labels[label_name]["pos"] = train_labels[label_name]["pos"]
    all_labels[label_name]["neg"] = train_labels[label_name]["neg"]
    print(f"Using train set for {label_name}. ")

    dataset = BinaryTask(label_dict=all_labels[label_name])
    for video, label in zip(dataset.videos, dataset.labels):
        if label == 1:
            label_str = "Yes"
            pos_samples += 1
        else:
            label_str = "No"
            neg_samples += 1
        all_samples.append({
            "video": video,
            "question": "{}".format(dataset.prompt),
            "answer": label_str
        })
        all_samples.append({
            "video": video,
            "question": "{} Please only answer Yes or No.".format(dataset.prompt),
            "answer": label_str
        })

print(f"Size of dataset: {len(all_samples)} with {pos_samples} positive samples and {neg_samples} negative samples")
import random
import json
random.shuffle(all_samples)
# with open(args.save_file, "w") as f:
#     json.dump(all_samples, f, indent=4)
# print(f"Saved to {args.save_file}")
