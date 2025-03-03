# Evaluate on camera-centric benchmark
# CUDA_VISIBLE_DEVICES=0 python cam_motion_pairwise_test_temp.py --score_model umt-b16-25m-clip
# CUDA_VISIBLE_DEVICES=1 python cam_motion_pairwise_test_temp.py --score_model umt-b16-25m-itm
# CUDA_VISIBLE_DEVICES=2 python cam_motion_pairwise_test_temp.py --score_model umt-l16-25m-clip
# CUDA_VISIBLE_DEVICES=3 python cam_motion_pairwise_test_temp.py --score_model umt-l16-25m-itm
# CUDA_VISIBLE_DEVICES=4 python cam_motion_pairwise_test_temp.py --score_model languagebind-video-v1.5-ft
# CUDA_VISIBLE_DEVICES=5 python cam_motion_pairwise_test_temp.py --score_model languagebind-video-ft
# CUDA_VISIBLE_DEVICES=6 python cam_motion_pairwise_test_temp.py --score_model internvideo2-1b-stage2-clip
# CUDA_VISIBLE_DEVICES=7 python cam_motion_pairwise_test_temp.py --score_model internvideo2-1b-stage2-itm
import argparse
import os
import sys
import t2v_metrics
import torch
from pathlib import Path

ROOT = Path("/data3/zhiqiul/video_annotation")
VIDEO_ROOT = ROOT / "videos"
VIDEO_LABEL_DIR = ROOT / "video_labels/"
# Get the absolute path of the video_annotation/ folder
# Add it to sys.path
sys.path.append(os.path.abspath(ROOT))

from pairwise_benchmark import (
    generate_pairwise_datasets,
    PairwiseBenchmark,
    print_detailed_task_statistics,
)


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
    "--train_ratio",
    type=float,
    help="The ratio of training samples",
    default=0.5
)
parser.add_argument(
    "--save_dir",
    type=str,
    help="The directory to save the results",
    default="./temp_pairwise_testset/",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="The batch size to use"
)
parser.add_argument(
    "--question",
    default="{} Please only answer Yes or No.",
    type=str
)
args = parser.parse_args()


print(f"Using score model: {args.score_model}")

score_model = t2v_metrics.get_score_model(model=args.score_model)

sampling_str = "top" if args.sampling == "top" else f"random_seed_{args.seed}"
folder_name = f"test_ratio_{1 - args.train_ratio:.2f}_num_{args.max_samples}_sampling_{sampling_str}"

question_str = args.question.replace(" ", "_")
SAVE_DIR = Path(args.save_dir) / folder_name / f"{args.score_model}_{question_str}"
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

datasets = generate_pairwise_datasets(
    max_samples=args.max_samples,
    sampling=args.sampling,
    seed=args.seed,
    root=args.root,
    video_root=args.video_root,
    video_labels_dir=args.video_label_dir,
    train_ratio=args.train_ratio
)

# Print statistics
print_detailed_task_statistics(
    datasets["original_tasks"]["raw"],
    datasets["sampled_tasks"]["train"],
    datasets["sampled_tasks"]["test"],
)

sampled_tasks = datasets["sampled_tasks"]["test"]

# if type of score_model is VQAScoreModel
if isinstance(score_model, t2v_metrics.VQAScore):
    benchmark = PairwiseBenchmark(sampled_tasks, mode="vqa")
    
    yes_kwargs = {"question_template": args.question, "answer_template": "Yes"}
    no_kwargs = {"question_template": args.question, "answer_template": "No"}
    
    yes_scores = score_model.batch_forward(benchmark, batch_size=args.batch_size, **yes_kwargs).cpu()
    no_scores = score_model.batch_forward(benchmark, batch_size=args.batch_size, **no_kwargs).cpu()
    
    # VQAScore Model can evaluate both retrieval and VQA
    retrieval_results, retrieval_tables = benchmark.evaluate_and_print_retrieval(yes_scores)
    vqa_results, vqa_tables = benchmark.evaluate_and_print_vqa(yes_scores, no_scores)
    
    # Save the results
    torch.save(retrieval_results, SAVE_DIR / "retrieval_results.pt")
    torch.save(vqa_results, SAVE_DIR / "vqa_results.pt")
    print(f"Saved results to {SAVE_DIR}/retrieval_results.pt")
    print(f"Saved results to {SAVE_DIR}/vqa_results.pt")
    
    # Save the tables in a text file
    with open(SAVE_DIR / "retrieval_tables.txt", "w") as f:
        f.write(retrieval_tables)
    with open(SAVE_DIR / "vqa_tables.txt", "w") as f:
        f.write(vqa_tables)
else:
    benchmark = PairwiseBenchmark(sampled_tasks, mode="retrieval")
    scores = score_model.batch_forward(benchmark, batch_size=args.batch_size).cpu()
    
    # CLIPScore/ITMScore Model can only evaluate retrieval
    retrieval_results, retrieval_tables = benchmark.evaluate_and_print_retrieval(scores)
    
    # Save the results
    torch.save(retrieval_results, SAVE_DIR / "retrieval_results.pt")
    print(f"Saved results to {SAVE_DIR}/retrieval_results.pt")
    
    # Save the tables in a text file
    with open(SAVE_DIR / "retrieval_tables.txt", "w") as f:
        f.write(retrieval_tables)
