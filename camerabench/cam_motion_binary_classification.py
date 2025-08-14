# Evaluate on camera-centric benchmark
# CUDA_VISIBLE_DEVICES=2 python cam_motion_better_question_test_temp.py --score_model umt-b16-25m-clip
# CUDA_VISIBLE_DEVICES=4 python cam_motion_better_question_test_temp.py --score_model umt-b16-25m-itm
# CUDA_VISIBLE_DEVICES=1 python cam_motion_better_question_test_temp.py --score_model languagebind-video-v1.5-ft
# CUDA_VISIBLE_DEVICES=2 python cam_motion_better_question_test_temp.py --score_model languagebind-video-ft
# CUDA_VISIBLE_DEVICES=3 python cam_motion_better_question_test_temp.py --score_model internvideo2-1b-stage2-clip
# CUDA_VISIBLE_DEVICES=4 python cam_motion_better_question_test_temp.py --score_model internvideo2-1b-stage2-itm
# CUDA_VISIBLE_DEVICES=5 python cam_motion_better_question_test_temp.py --score_model umt-l16-25m-clip
# CUDA_VISIBLE_DEVICES=6 python cam_motion_better_question_test_temp.py --score_model umt-l16-25m-itm
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


# from pairwise_benchmark import (
#     generate_pairwise_datasets,
# )
import argparse

# ROOT = Path("/data3/zhiqiul/video_annotation")
# VIDEO_ROOT = Path("/data3/zhiqiul/video_annotation/videos")
# VIDEO_LABEL_DIR = ROOT / "video_labels/"
# Get the absolute path of the video_annotation/ folder
# Add it to sys.path
sys.path.append(os.path.abspath(ROOT))

from benchmark import labels_as_dict, BinaryTask

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
    "--batch_size",
    type=int,
    default=64,
    help="The batch size to use"
)
# Just make using test set true to be default
# parser.add_argument(
#     "--use_testset",
#     action="store_true",
#     help="Use the test set for evaluation. Important: If this flag is true, then must use this flag for all subsequent runs."
# )
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

if 'gemini' in args.score_model:
    score_model = t2v_metrics.get_score_model(model=args.score_model, api_key='api_key')
elif 'gpt' in args.score_model:
    score_model = t2v_metrics.get_score_model(model=args.score_model, api_key='api_key')
else:
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

sampling_str = "top" if args.sampling == "top" else f"random_seed_{args.seed}"
folder_name = f"test_ratio_{1 - args.train_ratio:.2f}_num_{args.max_samples}_sampling_{sampling_str}"
video_label_str = "_".join(args.video_label_file.split("/")[-2:])[:-5]
use_testset_str = "_testset" #if args.use_testset else ""

# SAVE_DIR = Path(args.save_dir) / f"{video_label_str}_{folder_name}{use_testset_str}"
SAVE_DIR = Path(args.save_dir) / f"{video_label_str}"
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

all_labels = labels_as_dict(root=args.root, video_label_file=args.video_label_file, video_root=args.video_root)
scene_movement_labels = labels_as_dict(root=args.root, video_label_file=args.video_label_file_scene_movement, video_root=args.video_root)

label_to_name = {
    "cam_motion.camera_centric_movement.forward.has_forward_wrt_camera": "Move In",
    "cam_motion.camera_centric_movement.backward.has_backward_wrt_camera": "Move Out",
    "cam_motion.steadiness_and_movement.fixed_camera": "Static",
    "cam_motion.camera_centric_movement.zoom_in.has_zoom_in": "Zoom In",
    "cam_motion.camera_centric_movement.zoom_out.has_zoom_out": "Zoom Out",
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
# label_to_name = {
#     "cam_motion.steadiness_and_movement.fixed_camera": "Static",
#     "cam_motion.camera_centric_movement.zoom_in.has_zoom_in": "Zoom In",
#     "cam_motion.camera_centric_movement.zoom_out.has_zoom_out": "Zoom Out",
#     "cam_motion.camera_centric_movement.forward.has_forward_wrt_camera": "Move In",
#     "cam_motion.camera_centric_movement.backward.has_backward_wrt_camera": "Move Out",
#     "cam_motion.camera_centric_movement.upward.has_upward_wrt_camera": "Move Up",
#     "cam_motion.camera_centric_movement.downward.has_downward_wrt_camera": "Move Down",
#     "cam_motion.camera_centric_movement.rightward.has_rightward": "Move Right",
#     "cam_motion.camera_centric_movement.leftward.has_leftward": "Move Left",
#     "cam_motion.camera_centric_movement.pan_right.has_pan_right": "Pan Right",
#     "cam_motion.camera_centric_movement.pan_left.has_pan_left": "Pan Left",
#     "cam_motion.camera_centric_movement.tilt_up.has_tilt_up": "Tilt Up",
#     "cam_motion.camera_centric_movement.tilt_down.has_tilt_down": "Tilt Down",
#     "cam_motion.camera_centric_movement.roll_clockwise.has_roll_clockwise": "Roll Clockwise",
#     "cam_motion.camera_centric_movement.roll_counterclockwise.has_roll_counterclockwise": "Roll Counterclockwise",
#     "cam_motion.scene_movement.dynamic_scene": "Dynamic Scene",
#     "cam_motion.scene_movement.mostly_static_scene": "Mostly Static Scene",
#     "cam_motion.scene_movement.static_scene": "Static Scene",
# }

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

print(f"Loaded {len(all_labels)} labels and {len(scene_movement_labels)} scene movement labels")
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

all_labels_scores = {}
for label_name in all_labels:
    LABEL_SAVE_DIR = SAVE_DIR / label_name
    if not LABEL_SAVE_DIR.exists():
        LABEL_SAVE_DIR.mkdir(parents=True)
    LABEL_SAVE_PATH = LABEL_SAVE_DIR / f"{args.score_model}_scores{use_testset_str}.pt"

    # if args.use_testset:
    all_labels[label_name]["pos"] = test_labels[label_name]["pos"]
    all_labels[label_name]["neg"] = test_labels[label_name]["neg"]
    print(f"Using test set for {label_name}. If you want to use the full set, please rerun without the --use_testset flag.")
    
    dataset = BinaryTask(label_dict=all_labels[label_name])
    # print(dataset[:5])
    # exit()

    if LABEL_SAVE_PATH.exists():
        print(f"Already finished {label_name}")
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
            plot_path=LABEL_SAVE_DIR / f"{args.score_model}_pr_curve{use_testset_str}.jpeg",
        )
        all_labels_scores[label_name] = {
            "scores": scores,
            "results": results,
        }
        torch.save(all_labels_scores[label_name], LABEL_SAVE_PATH)


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


# header = f"{args.score_model:80s} {'AP':>10} {'ROC AUC':>10} {'F1':>10} {'Threshold':>12}"
# separator = "-" * len(header)
subset_names = [
    "Dynamic Scene Balanced (Test)",
    "Dynamic Scene Balanced (All)",
    "Mostly Static Scene Balanced (Test)",
    "Mostly Static Scene Balanced (All)",
    "Static Scene Balanced (Test)",
    "Static Scene Balanced (All)",
    "Dynamic and Mostly Static Scene Balanced (Test)",
    "Dynamic and Mostly Static Scene Balanced (All)",
    "Mostly Static and Static Scene Balanced (Test)",
    "Mostly Static and Static Scene Balanced (All)",
    
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
    # print(header)
    # print(separator)
    # ap = scores["results"]["ap"]
    # roc_auc = scores["results"]["roc_auc"]
    # optimal_f1 = scores["results"]["optimal_f1"]
    # optimal_threshold = scores["results"]["optimal_threshold"]
    label_name_nice = label_to_name[label_name]
    # print(
    #     f"{label_name_nice:80s} {ap:>10.4f} {roc_auc:>10.4f} {optimal_f1:>10.4f} {optimal_threshold:>12.4f}"
    # )
    # if label_name_nice == "Static":
    #     print(f"We didn't label the scene movement for {label_name_nice}.")
    #     continue
    subsets = {
        # "Dynamic Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Dynamic Scene"]),
        # "Mostly Static Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Mostly Static Scene"]),
        # "Static Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Static Scene"]),
        # "Dynamic and Mostly Static Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Dynamic Scene", "Mostly Static Scene"]),
        # "Mostly Static and Static Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=["Mostly Static Scene", "Static Scene"]),
        "All Scene (Test)": get_subset_videos(label_name, split="test", scene_movement=[]),
        "Dynamic Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Dynamic Scene"], balanced=True),
        "Mostly Static Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Mostly Static Scene"], balanced=True),
        "Static Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Static Scene"], balanced=True),
        "Dynamic and Mostly Static Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Dynamic Scene", "Mostly Static Scene"], balanced=True),
        "Mostly Static and Static Scene Balanced (Test)": get_subset_videos(label_name, split="test", scene_movement=["Mostly Static Scene", "Static Scene"], balanced=True),

        # "Mostly Static Scene (All)": get_subset_videos(label_name, split="all", scene_movement=["Mostly Static Scene"]),
        # "Static Scene (All)": get_subset_videos(label_name, split="all", scene_movement=["Static Scene"]),
        # "Dynamic and Mostly Static Scene (All)": get_subset_videos(label_name, split="all", scene_movement=["Dynamic Scene", "Mostly Static Scene"]),
        # "Mostly Static and Static Scene (All)": get_subset_videos(label_name, split="all", scene_movement=["Mostly Static Scene", "Static Scene"]),
        "All Scene (All)": get_subset_videos(label_name, split="all", scene_movement=[]),
        "Dynamic Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Dynamic Scene"], balanced=True),
        "Mostly Static Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Mostly Static Scene"], balanced=True),
        "Static Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Static Scene"], balanced=True),
        "Dynamic and Mostly Static Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Dynamic Scene", "Mostly Static Scene"], balanced=True),
        "Mostly Static and Static Scene Balanced (All)": get_subset_videos(label_name, split="all", scene_movement=["Mostly Static Scene", "Static Scene"], balanced=True),

    }
    dataset = BinaryTask(label_dict=all_labels[label_name])
    raw_scores = scores["scores"]
    for subset_name, subset_videos in subsets.items():
        print_name = f"{label_name_nice} for {subset_name}"
        if subset_name not in ["All Scene (All)", "All Scene (Test)"] and label_name_nice == "Static":
            # print(f"Skipping {subset_name} for {label_name_nice}")
            continue
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
