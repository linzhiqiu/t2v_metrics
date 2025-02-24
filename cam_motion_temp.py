# Evaluate on all datasets in VQAScore paper

import argparse
import os
import sys
import t2v_metrics
import torch
from torch.utils.data import Dataset
import pathlib
from pathlib import Path

ROOT = Path("/data3/zhiqiul/video_annotation")
VIDEO_ROOT = Path("/data3/zhiqiul/video_annotation/videos")
# Get the absolute path of the video_annotation/ folder
# Add it to sys.path
sys.path.append(os.path.abspath(ROOT))

from download import get_labels
from label import extract_labels_dict, Label

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
    "--video_label_file",
    type=str,
    help="The video label to use",
    # default="video_labels/cam_motion-20250219_0338/label_names.json", @ noisy 4000
    # default="video_labels/cam_motion-cam_setup-20250218_1042/label_names.json", # good initial set of around 2000
    # default="video_labels/cam_motion-20250223_0313/label_names_subset.json", # only a subset of 300 videos
    default="video_labels/cam_motion-20250223_2308/label_names_selected.json", # the full set of 2000 videos for cam-centric
    # default="video_labels/cam_motion-cam_setup-20250224_0130/label_names_selected.json", # the full set of 2384 videos for ground-centric + shotcomp
    nargs="?",
)
parser.add_argument("--batch_size", type=int, default=256, help="The batch size to use")
args = parser.parse_args()

score_model = args.score_model
print(f"Using score model: {score_model}")

umt_score = t2v_metrics.get_score_model(model=score_model)


ALL_LABELS = get_labels(ROOT / args.video_label_file)
video_label_str = "_".join(args.video_label_file.split("/")[-2:])[:-5]

SAVE_DIR = Path(f"./temp/") / f"{video_label_str}" 
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)
LABEL_SAVE_PATH = SAVE_DIR / "labels.pt"
if LABEL_SAVE_PATH.exists():
    labels_collection = torch.load(LABEL_SAVE_PATH)
else:
    labels_collection = extract_labels_dict(Label.load_all_labels(labels_dir=ROOT / "labels"))

class BinaryTask(Dataset):

    def __init__(
        self,
        video_dir=VIDEO_ROOT,
        label_name="cam_setup.angle.camera_angle_change_from_high_to_low",
    ):
        self.video_dir = video_dir
        self.pos = [video_dir / video_path for video_path in ALL_LABELS[label_name]['pos']]
        self.neg = [video_dir / video_path for video_path in ALL_LABELS[label_name]['neg']]
        self.labels = [1] * len(self.pos) + [0] * len(self.neg)
        self.videos = self.pos + self.neg
        self.def_prompts = labels_collection[label_name].def_prompt
        self.alt_prompts = labels_collection[label_name].alt_prompt
        self.prompts = self.def_prompts + self.alt_prompts
        self.label_name = label_name

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        item = {"images": [str(video_path)], "texts": self.prompts}
        return item

    def evaluate_scores(self, scores, plot_path=None):
        # Calculate the Average Precision for each prompt
        from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score
        import numpy as np
        print(f"Evaluating for task {self.label_name}")
        y_true = self.labels
        results = {}
        for prompt_idx, prompt in enumerate(self.prompts):
            y_scores = scores[:, 0, prompt_idx].cpu().numpy()
            ap = average_precision_score(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
            # Compute Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

            # Compute F1 scores for all thresholds
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
            best_idx = np.argmax(f1_scores)
            optimal_f1 = f1_scores[best_idx]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5  # Fallback threshold

            # print(f"Prompt ({'def' if is_def else 'alt'}): {prompt}")
            # print(f"AP: {ap}, ROC AUC: {roc_auc}")
            results[prompt] = {
                "ap": float(ap), "roc_auc": float(roc_auc),
                "optimal_f1": float(optimal_f1), "optimal_threshold": float(optimal_threshold)
            }
        # print highest performing prompt
        best_prompt = max(results, key=lambda x: results[x]["ap"])
        best_prompt_index = self.prompts.index(best_prompt)
        best_prompt_str = f"{best_prompt} ({'def' if best_prompt in self.def_prompts else 'alt'})"
        print(f"Best prompt: {best_prompt_str:100s} AP: {results[best_prompt]['ap']:.4f}, ROC AUC: {results[best_prompt]['roc_auc']:.4f}, Optimal F1: {results[best_prompt]['optimal_f1']:.4f}")
        results["best"] = {
            "prompt": best_prompt,
            "ap": results[best_prompt]["ap"],
            "roc_auc": results[best_prompt]["roc_auc"],
            "optimal_f1": results[best_prompt]["optimal_f1"],
            "optimal_threshold": results[best_prompt]["optimal_threshold"]
        }
        if plot_path:
            print(f"Plotting to {plot_path}")
            import matplotlib.pyplot as plt
            y_scores = scores[:, 0, best_prompt_index].cpu().numpy()
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_precision, best_recall = precision[best_idx], recall[best_idx]
            optimal_f1 = f1_scores[best_idx]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

            plt.figure(figsize=(8, 6))

            # Plot Precision-Recall Curve
            plt.plot(recall, precision, label=f'Precision-Recall Curve (AP={results[best_prompt]["ap"]:.4f})', color='blue', linewidth=2)

            # Highlight the best threshold point
            plt.scatter(best_recall, best_precision, color='red', zorder=3, label=f'Optimal F1={optimal_f1:.4f} @ Threshold={optimal_threshold:.2f}', s=100)

            # Labels and Title
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)
            plt.title(f"Precision-Recall Curve\n{best_prompt_str}", fontsize=14)
            plt.legend(loc="lower left", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.ylim([0, 1])
            # Save the plot
            plt.savefig(plot_path, bbox_inches="tight", dpi=300)
            plt.close()
        return results


# Sort all labels by the total number of positive and negative examples
# label_counts = {}
# for label_name in ALL_LABELS:
#     pos = len(ALL_LABELS[label_name]['pos'])
#     neg = len(ALL_LABELS[label_name]['neg'])
#     label_counts[label_name] = pos + neg

# for label_name in ALL_LABELS:
#     print(f"{label_name}: Positives: {len(ALL_LABELS[label_name]['pos'])}, Negatives: {len(ALL_LABELS[label_name]['neg'])}")
# sorted_labels = sorted(label_counts, key=lambda x: label_counts[x])
# for label_name in sorted_labels[:10]:
#     print(f"{label_name}: {label_counts[label_name]}")
# import pdb; pdb.set_trace()

# dataset = BinaryTask()
# # scores = umt_score.batch_forward(dataset, batch_size=16)
# scores = torch.FloatTensor(len(dataset.videos), 1, len(dataset.prompts)).uniform_(-1, 1)
# results = dataset.evaluate_scores(scores)

SAVE_PATH = Path(SAVE_DIR / f"{score_model}_scores.pt")

if SAVE_PATH.exists():
    saved_results = torch.load(SAVE_PATH)
    all_labels_scores = saved_results["all_labels_scores"]
    all_labels = saved_results["all_labels"]
else:
    
    all_labels_scores = {}
    for label_name in ALL_LABELS:
        LABEL_SAVE_DIR = SAVE_DIR / label_name
        if not LABEL_SAVE_DIR.exists():
            LABEL_SAVE_DIR.mkdir(parents=True)
        LABEL_SAVE_PATH = LABEL_SAVE_DIR / f"{score_model}_scores.pt"
        dataset = BinaryTask(label_name=label_name)
        if LABEL_SAVE_PATH.exists():
            print(f"Skipping {label_name}")
            all_labels_scores[label_name] = torch.load(LABEL_SAVE_PATH)
            results = dataset.evaluate_scores(all_labels_scores[label_name]["scores"], plot_path=LABEL_SAVE_DIR / f"{score_model}_pr_curve.jpeg")
            continue
        scores = umt_score.batch_forward(dataset, batch_size=args.batch_size)
        results = dataset.evaluate_scores(scores)
        all_labels_scores[label_name] = {
            "scores": scores,
            "results": results,
            "best_prompt": results["best"]["prompt"],
            "best_ap": results["best"]["ap"]
        }
        torch.save(all_labels_scores[label_name], LABEL_SAVE_PATH)
    # torch.save(
    #     {
    #         "all_labels_scores": all_labels_scores,
    #         "all_labels": all_labels
    #     },
    #     SAVE_PATH
    # )

# Sort by best AP
sorted_labels = sorted(all_labels_scores, key=lambda x: all_labels_scores[x]["best_ap"], reverse=True)
for label_name in sorted_labels:
    print(f"{label_name:50s}: Best AP: {all_labels_scores[label_name]['best_ap']:.4f} with prompt {all_labels_scores[label_name]['best_prompt']:100s}")
