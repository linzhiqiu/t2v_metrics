# Evaluate on camera-centric benchmark
# python cam_motion_sfm.py --score_model megasam_posed_videos
# python cam_motion_sfm.py --score_model cut3r_posed_videos
# python cam_motion_sfm.py --score_model posed_vggsfm_videos
# python cam_motion_sfm.py --score_model posed_colmap_videos
# python cam_motion_sfm.py --score_model posed_mast3r_videos
# python cam_motion_sfm.py --score_model megasam_extracted_videos_4_fps
# python cam_motion_sfm.py --score_model megasam_extracted_videos_8_fps
import argparse
import os
import sys
import torch
import json
from torch.utils.data import Dataset
from pathlib import Path
import random
import shutil

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
    # default="video_labels/cam_motion-20250219_0338/label_names.json", @ noisy 4000
    # default="video_labels/cam_motion-cam_setup-20250218_1042/label_names.json", # good initial set of around 2000
    # default="video_labels/cam_motion-20250223_0313/label_names_subset.json", # only a subset of 300 videos
    # default="video_labels/cam_motion-20250223_2308/label_names_selected.json", # the full set of 2000 videos for cam-centric
    # default="video_labels/cam_motion-cam_setup-20250224_0130/label_names_selected.json", # the full set of 2384 videos for ground-centric + shotcomp
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
    "--save_dir",
    type=str,
    help="The directory to save the results",
    default="./temp/",
)
parser.add_argument(
    "--sfm_dir",
    type=str,
    help="The directory to load the sfm data",
    default="sfm_outputs/",
)
parser.add_argument(
    "--score_model",
    type=str,
    help="The sfm method to use",
    # default="megasam_posed_videos",
    default="posed_colmap_videos",
    # default="posed_vggsfm_videos",
    # default="cut3r_posed_videos",
)
args = parser.parse_args()


video_label_str = "_".join(args.video_label_file.split("/")[-2:])[:-5]

SAVE_DIR = Path(args.save_dir) / f"{video_label_str}" 
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

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

# print(f"Loaded {len(all_labels)} labels")
# label_counts = {}
# for label_name in all_labels:
#     pos = len(all_labels[label_name]['pos'])
#     neg = len(all_labels[label_name]['neg'])
#     label_counts[label_name] = pos + neg

# for label_name in all_labels:
#     print(
#         f"{all_labels[label_name]['label_name']:100s}: Positives: {len(all_labels[label_name]['pos']):10d}, Negatives: {len(all_labels[label_name]['neg']):10d}"
#     )
#     # Random chance AP is the ratio of positives to total samples
#     print(
#         f"Random Chance AP: {len(all_labels[label_name]['pos']) / label_counts[label_name]:.4f}"
#     )


def move_and_remove_cut3r_out(base_folder):
    # Iterate through all subdirectories in the base folder
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        cut3r_out_path = os.path.join(subfolder_path, "cut3r_out")

        # Ensure it's a directory and contains 'cut3r_out'
        if os.path.isdir(subfolder_path) and os.path.exists(cut3r_out_path):
            # print(f"Processing: {subfolder_path}")

            # Move all files from cut3r_out to the parent folder
            for item in os.listdir(cut3r_out_path):
                src_path = os.path.join(cut3r_out_path, item)
                dest_path = os.path.join(subfolder_path, item)

                if os.path.exists(dest_path):
                    # print(f"Skipping {item}, already exists in {subfolder_path}")
                    continue

                shutil.copy(src_path, dest_path)

            # Remove the empty cut3r_out folder
            # os.rmdir(cut3r_out_path)
            # print(f"Moved contents {cut3r_out_path}")


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

print(
    f"Loaded {len(all_labels)} labels and {len(scene_movement_labels)} scene movement labels"
)
label_counts = {}
test_labels = {}
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
    pos = all_labels[label_name]["pos"]
    neg = all_labels[label_name]["neg"]
    label_counts[label_name] = {"pos": pos, "neg": neg, "total": pos + neg}
    pos_test = [v for v in all_labels[label_name]["pos"] if v in test_videos]
    neg_test = [v for v in all_labels[label_name]["neg"] if v in test_videos]
    test_labels[label_name] = {
        "pos": pos_test,
        "neg": neg_test,
        "total": pos_test + neg_test,
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
            "total": pos_scene + neg_scene,
        }

    pos_to_neg_ratio = []
    pos_to_neg_ratio_test = []
    for scene_name in scene_labels_balanced:
        pos_to_neg_ratio.append(
            len(scene_labels[scene_name][label_name]["pos"])
            / len(scene_labels[scene_name][label_name]["neg"])
        )
        pos_to_neg_ratio_test.append(
            len(scene_labels[scene_name][label_name]["pos_test"])
            / len(scene_labels[scene_name][label_name]["neg_test"])
        )
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
            "total": pos_scene_balanced + neg_scene_balanced,
        }

for label_name in all_labels:
    pos = label_counts[label_name]["pos"]
    neg = label_counts[label_name]["neg"]
    pos_test = test_labels[label_name]["pos"]
    neg_test = test_labels[label_name]["neg"]
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
        print(
            f"{'':25s}{scene_name:25s}: Positives (All/Test/Total): {len(pos_scene):5d}/{len(pos_scene_test):5d}/{len(pos):5d}, Negatives (All/Test/Total): {len(neg_scene):5d}/{len(neg_scene_test):5d}/{len(neg):5d}"
        )
        print(
            f"{'':25s}{scene_name+' Balanced':25s}: Positives (All/Test/Total): {len(pos_scene_balanced):5d}/{len(pos_scene_balanced_test):5d}/{len(pos_scene_balanced):5d}, Negatives (All/Test/Total): {len(neg_scene_balanced):5d}/{len(neg_scene_balanced_test):5d}/{len(neg_scene_balanced):5d}"
        )


def combine_pose_diff_files(parent_dir):
    parent_path = Path(parent_dir)
    combined_data = {}

    # Iterate through all subdirectories
    for subfolder in [f for f in parent_path.iterdir() if f.is_dir()]:
        video_name = subfolder.name
        json_path = subfolder / "pose_diff.json"

        # Check if pose_diff.json exists in this subfolder
        if json_path.exists():
            try:
                # Load the JSON file
                with open(json_path, "r") as file:
                    json_data = json.load(file)

                # Add to the combined dictionary with video name as key
                combined_data[video_name] = json_data
                # print(f"Loaded {video_name}/pose_diff.json")
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON in {json_path}")
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    print(f"Total videos processed: {len(combined_data)}")
    return combined_data


# def save_combined_data(combined_data, output_file="combined_pose_data.json"):
#     with open(output_file, "w") as file:
#         json.dump(combined_data, file, indent=2)

#     print(f"Combined data saved to {output_file}")


parent_directory = Path(args.sfm_dir) / args.score_model

if args.score_model == "cut3r_posed_videos":
    move_and_remove_cut3r_out(parent_directory)

# Combine all pose_diff.json files
combined_data = combine_pose_diff_files(parent_directory)

# example of Jay's output {
#     "delta_x": -0.5157333612442017,
#     "delta_y": -0.1708381623029709,
#     "delta_z": 0.1114729717373848,
#     "delta_yaw": -0.12438342533492158,
#     "delta_pitch": 0.60178686397772,
#     "delta_roll": -0.22627989475546217,
#     "delta_zoom": 1.0,
# }

camera_movement_mapping = {
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
}

def static_score(item):
    return -max(
        abs(item["delta_x"]),
        abs(item["delta_y"]),
        abs(item["delta_z"]),
        abs(item["delta_yaw"] / 180),
        abs(item["delta_pitch"] / 180),
        abs(item["delta_roll"] / 180),
        abs(item["delta_zoom"] - 1.0),
    )

motion_to_score = {
    "Static": lambda item: static_score(item),
    "Zoom In": lambda item: item["delta_zoom"],
    "Zoom Out": lambda item: -item["delta_zoom"],
    "Move In": lambda item: item["delta_z"],
    "Move Out": lambda item: -item["delta_z"],
    "Move Up": lambda item: -item["delta_y"],
    "Move Down": lambda item: item["delta_y"],
    "Move Right": lambda item: item["delta_x"],
    "Move Left": lambda item: -item["delta_x"],
    "Pan Right": lambda item: item["delta_pitch"],
    "Pan Left": lambda item: -item["delta_pitch"],
    "Tilt Up": lambda item: item["delta_roll"],
    "Tilt Down": lambda item: -item["delta_roll"],
    "Roll Clockwise": lambda item: item["delta_yaw"],
    "Roll Counterclockwise": lambda item: -item["delta_yaw"],
}

# def static_score(item):
#     return max()

# print all value names
print(camera_movement_mapping.values())

raw_scores = [
    "delta_x"
    "delta_y"
    "delta_z"
    "delta_yaw"
    "delta_pitch"
    "delta_roll"
    "delta_zoom"
]

all_labels_scores = {}
missing_videos = []
for label_name in all_labels:
    LABEL_SAVE_DIR = SAVE_DIR / label_name
    if not LABEL_SAVE_DIR.exists():
        LABEL_SAVE_DIR.mkdir(parents=True)
    LABEL_SAVE_PATH = LABEL_SAVE_DIR / f"{args.score_model}_scores.pt"

    dataset = BinaryTask(label_dict=all_labels[label_name])

    videos = [item["images"][0] for item in dataset]
    video_names = []
    for video in videos:
        video_name = Path(video).stem
        video_names.append(video_name)

    video_items = []
    for video_name in video_names:
        if video_name not in combined_data:
            # print(f"Video {video_name} not found in pose data")
            video_items.append(None)
            continue
        video_items.append(combined_data[video_name])

    if not all([video_name in combined_data for video_name in video_names]):
        print(f"Missing video number is {len([video_name for video_name in video_names if video_name not in combined_data])}")
        missing_videos += [video_name + ".mp4" for video_name in video_names if video_name not in combined_data]
        import pdb; pdb.set_trace()
    
    import numpy as np
    label_name_nice = camera_movement_mapping[label_name]
    scores = []
    for item in video_items:
        if item is None:
            scores.append(np.nan)
            continue
        scores.append(motion_to_score[label_name_nice](item))
        
    scores = np.array(scores)

    results = dataset.evaluate_scores(
        scores,
        plot_path=LABEL_SAVE_DIR / f"{args.score_model}_pr_curve.jpeg",
    )
    all_labels_scores[label_name] = {
        "scores": scores,
        "results": results,
    }
    # torch.save(all_labels_scores[label_name], LABEL_SAVE_PATH)


# print by nice names:
print("Label Name".ljust(50), "AP".rjust(10), "ROC AUC".rjust(10), "F1".rjust(10), "Threshold".rjust(12))
print("-" * 100)
for label_name in all_labels:
    label_name_nice = camera_movement_mapping[label_name]
    ap = all_labels_scores[label_name]["results"]["ap"]
    roc_auc = all_labels_scores[label_name]["results"]["roc_auc"]
    optimal_f1 = all_labels_scores[label_name]["results"]["optimal_f1"]
    optimal_threshold = all_labels_scores[label_name]["results"]["optimal_threshold"]

    print(
        f"{label_name_nice:50s} {ap:>10.4f} {roc_auc:>10.4f} {optimal_f1:>10.4f} {optimal_threshold:>12.4f}"
    )

missing_videos = list(set(missing_videos))
with open("missing_videos.txt", "w") as f:
    f.write("\n".join(missing_videos))
print(f"Missing videos saved to 'missing_videos.txt'")

def get_subset_videos(label_name, split="all", balanced=False, scene_movement=["Dynamic Scene", "Mostly Static Scene", "Static Scene"]):
    if split == "all":
        all_videos = all_labels[label_name]["pos"] + all_labels[label_name]["neg"]
        pos_name = "pos"
        neg_name = "neg"
    elif split == "test":
        all_videos = test_labels[label_name]["pos"] + test_labels[label_name]["neg"]
        pos_name = "pos_test"
        neg_name = "neg_test"
    else:
        raise ValueError(f"Invalid split: {split}")

    if len(scene_movement) == 0:
        scene_videos = all_videos
    else:
        scene_videos = []
        for scene_name in scene_movement:
            if balanced:
                scene_videos += scene_labels_balanced[scene_name][label_name][pos_name] + scene_labels_balanced[scene_name][label_name][neg_name]
            else:
                scene_videos += scene_labels[scene_name][label_name][pos_name] + scene_labels[scene_name][label_name][neg_name]
    # import pdb; pdb.set_trace()
    return list(set(all_videos).intersection(set(scene_videos)))


subset_names = [
    # "Dynamic Scene Balanced (Test)",
    "Dynamic Scene Balanced (All)",
    # "Mostly Static Scene Balanced (Test)",
    "Mostly Static Scene Balanced (All)",
    # "Static Scene Balanced (Test)",
    "Static Scene Balanced (All)",
    # "Dynamic and Mostly Static Scene Balanced (Test)",
    # "Dynamic and Mostly Static Scene Balanced (All)",
    # "Mostly Static and Static Scene Balanced (Test)",
    # "Mostly Static and Static Scene Balanced (All)",
    "All Scene (Test)",
    "All Scene (All)",
]

paper_names = {
    'Move In': 'In',
    'Move Out': 'Out',
    'Move Up': 'Up',
    'Move Down': 'Down',
    'Move Right': 'Right',
    'Move Left': 'Left',
    'Zoom In': 'Zoom In',
    'Zoom Out': 'Zoom Out',
    'Pan Right': 'Pan Right',
    'Pan Left': 'Pan Left',
    'Tilt Up': 'Tilt Up',
    'Tilt Down': 'Tilt Down',
    'Roll Clockwise': 'Roll CW',
    'Roll Counterclockwise': 'Roll CCW',
    'Static': 'Static',
}

from copy import deepcopy
scene_analysis_names = deepcopy(paper_names)
# Remove Static
del scene_analysis_names["Static"]


all_subset_results = {key: {} for key in subset_names}

for label_name, scores in all_labels_scores.items():
    label_name_nice = label_to_name[label_name]
    subsets = {
        # "Dynamic Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Dynamic Scene"]),
        # "Mostly Static Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Mostly Static Scene"]),
        # "Static Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Static Scene"]),
        # "Dynamic and Mostly Static Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Dynamic Scene", "Mostly Static Scene"]),
        # "Mostly Static and Static Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Mostly Static Scene", "Static Scene"]),
        # "Dynamic Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Dynamic Scene"], balanced=True),
        # "Mostly Static Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Mostly Static Scene"], balanced=True),
        # "Static Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Static Scene"], balanced=True),
        "All Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=[]),
        # "Dynamic and Mostly Static Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Dynamic Scene", "Mostly Static Scene"], balanced=True),
        # "Mostly Static and Static Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Mostly Static Scene", "Static Scene"], balanced=True),

        "Dynamic Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Dynamic Scene"], balanced=True),
        # "Mostly Static Scene (All)": get_subset_videos(label_name, split="all", scene_movement=["Mostly Static Scene"]),
        # "Static Scene (All)": get_subset_videos(label_name, split="all", scene_movement=["Static Scene"]),
        # "Dynamic and Mostly Static Scene (All)": get_subset_videos(label_name, split="all", scene_movement=["Dynamic Scene", "Mostly Static Scene"]),
        # "Mostly Static and Static Scene (All)": get_subset_videos(label_name, split="all", scene_movement=["Mostly Static Scene", "Static Scene"]),
        "Static Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Static Scene"], balanced=True),
        "Mostly Static Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Mostly Static Scene"], balanced=True),
        "All Scene (All)": get_subset_videos(label_name, split="all", scene_movement=[]),
        # "Dynamic and Mostly Static Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Dynamic Scene", "Mostly Static Scene"], balanced=True),
        # "Mostly Static and Static Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Mostly Static Scene", "Static Scene"], balanced=True),

    }
    dataset = BinaryTask(label_dict=all_labels[label_name])
    raw_scores = scores["scores"]
    for subset_name, subset_videos in subsets.items():
        print_name = f"{label_name_nice} for {subset_name}"
        if subset_name not in ["All Scene (All)", "All Scene (Test)"] and label_name_nice == "Static":
            # print(f"Skipping {subset_name} for {label_name_nice}")
            continue
        # import pdb; pdb.set_trace()
        results = dataset.evaluate_scores_subset(
            raw_scores,
            subset_videos,
            score_name=args.score_model,
            print_name=print_name,
            save_path=LABEL_SAVE_DIR / f"{args.score_model}_{subset_name}.txt",
        )
        all_subset_results[subset_name][label_name_nice] = {
            "ap": results["ap"],
            "random_ap": results["random_ap"],
        }
        print()

# Print the results for each subset and all labels according to the paper names
for subset_name in subset_names:
    if subset_name in ["All Scene (All)", "All Scene (Test)"]:
        used_names = paper_names
    else:
        used_names = scene_analysis_names
    all_aps = [
        all_subset_results[subset_name][label_name_nice]["ap"] * 100.
        for label_name_nice in used_names
    ]
    all_random_aps = [
        all_subset_results[subset_name][label_name_nice]["random_ap"] * 100.
        for label_name_nice in used_names
    ]
    mean_ap = sum(all_aps) / len(all_aps)
    mean_random_ap = sum(all_random_aps) / len(all_random_aps)
    all_aps.append(mean_ap)
    all_random_aps.append(mean_random_ap)
    used_names = deepcopy(list(used_names.values())) + ["Avg"]
    print(f"{subset_name:40s}" + " ".join([name.rjust(10) for name in used_names]))
    print("-" * 40 + " " + "-" * 11 * len(used_names))
    print(f"{'Random AP':40s}" + " ".join([f"{ap:10.1f}" for ap in all_random_aps]))
    print(f"{'AP':40s}" + " ".join([f"{ap:10.1f}" for ap in all_aps]))
    print()
