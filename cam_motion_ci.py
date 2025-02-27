# Evaluate on all datasets in VQAScore paper
# Model List = [gemini-2.0, gemini-1.5-flash, gpt4v, gpt-4o, internlm-7b, internvideo2-chat-8b, internvl2.5 7B, internvl2.5-26B, llava-ov-7B, llava-video-7b, mplug-7b, qwen-2-7b, qwen2.57b, tarsier, blip2_item, image_reward, internvideo2_itm, umt_itm, internvideo2_clip, languagebind_clip, umt_clip]
import argparse
import os
import sys
import t2v_metrics
import torch
from torch.utils.data import Dataset
import pathlib
from pathlib import Path
import random
import gc

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
    default="languagebind-video-v1.5-ft",
    # default="languagebind-video-v1.5-huge-ft",
    help="The score model to use",
)
parser.add_argument(
    "--video_label_file",
    type=str,
    help="The video label to use",
    default="video_labels/cam_motion-20250223_2308/label_names_selected.json", # the full set of 2000 videos for cam-centric
    # default="video_labels/cam_motion-cam_setup-20250224_0130/label_names_selected.json", # the full set of 2384 videos for ground-centric + shotcomp
    nargs="?",
)
parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use")
parser.add_argument("--question", default=None, type=str)
parser.add_argument("--answer", default=None, type=str)
parser.add_argument("--mode", default="vqa", type=str, choices=["vqa", "retrieval"])
args = parser.parse_args()

score_model = args.score_model
print(f"Using score model: {score_model}")

score_model = t2v_metrics.get_score_model(model=score_model)


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
        self.def_questions = labels_collection[label_name].def_question
        self.alt_questions = labels_collection[label_name].alt_question
        self.prompts = self.def_questions + self.alt_questions
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
        best_prompt_str = f"{best_prompt}"
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


class HasForward(BinaryTask):

    def __init__(
        self,
        video_dir=VIDEO_ROOT,
        label_name="cam_motion.camera_centric_movement.forward.has_forward_wrt_camera",
        seed=42,
        num_pos_samples=100,
        mode="vqa",
    ):
        self.video_dir = video_dir
        self.seed = seed
        pos_videos = [
            video_dir / video_path for video_path in ALL_LABELS[label_name]["pos"]
        ]
        neg_videos = [
            video_dir / video_path for video_path in ALL_LABELS[label_name]["neg"]
        ]
        random.seed(self.seed)
        # Sample 100 positive videos (or all if fewer than 100 exist)
        self.pos = random.sample(pos_videos, min(num_pos_samples, len(pos_videos)))
        self.neg = neg_videos  # Keep all negatives

        self.labels = [1] * len(self.pos) + [0] * len(self.neg)
        self.videos = self.pos + self.neg
        if mode == "vqa":
            self.defs = labels_collection[label_name].def_question
        else:
            self.defs = labels_collection[label_name].def_prompt
        self.prompts = [self.defs[0]]
        self.label_name = label_name


kwargs = {}
if args.question is not None:
    print(f"Using question template: {args.question}")
    kwargs['question_template'] = args.question
if args.answer is not None:
    print(f"Using answer template: {args.answer}")
    kwargs['answer_template'] = args.answer

label_name = "cam_motion.camera_centric_movement.forward.has_forward_wrt_camera"
LABEL_SAVE_DIR = SAVE_DIR / label_name
if not LABEL_SAVE_DIR.exists():
    LABEL_SAVE_DIR.mkdir(parents=True)
LABEL_SAVE_PATH = LABEL_SAVE_DIR / f"{args.score_model}_scores_random_seed.pt"
all_seeds_results = {}
if LABEL_SAVE_PATH.exists():
    print(f"already exists: {LABEL_SAVE_PATH}")
    print(f"Skipping {label_name}")
    all_seeds_results = torch.load(LABEL_SAVE_PATH)
else:
    import time
    time_start = time.time()
    for seed in range(42, 52):
        dataset = HasForward(label_name=label_name, seed=seed)
        print(f'Batch Forward Step')
        scores = score_model.batch_forward(
            dataset,
            batch_size=args.batch_size,
            **kwargs
        )
        results = dataset.evaluate_scores(scores)
        all_seeds_results[seed] = {
            "scores": scores,
            "results": results,
            "best_prompt": results["best"]["prompt"],
            "best_ap": results["best"]["ap"],
        }
        # Add memory cleanup here
        torch.cuda.empty_cache()
        gc.collect()
    torch.save(all_seeds_results, LABEL_SAVE_PATH)
    time_end = time.time()
    time_total = time_end - time_start
    # print time in minutes
    print(f"Time taken: {time_total / 60:.2f} minutes")

# Print the best AP for each seed, and show the
import numpy as np
import scipy.stats as stats

# Extract best_ap values from the dictionary
best_ap_values = [data["best_ap"] for data in all_seeds_results.values()]

# Compute mean and standard deviation
mean_ap = np.mean(best_ap_values)
std_ap = np.std(best_ap_values, ddof=1)  # Use ddof=1 for sample standard deviation

# Compute 95% confidence interval using normal approximation
z_score = stats.norm.ppf(0.975)  # 1.96 for 95% CI
ci_lower = mean_ap - z_score * (std_ap / np.sqrt(len(best_ap_values)))
ci_upper = mean_ap + z_score * (std_ap / np.sqrt(len(best_ap_values)))

# Print results
print(f"Mean AP: {mean_ap:.3f}")
print(f"Std Dev: {std_ap:.3f}")
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
