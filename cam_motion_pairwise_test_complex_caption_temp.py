# Evaluate on camera-centric benchmark
# CUDA_VISIBLE_DEVICES=0 python cam_motion_pairwise_test_complex_caption_temp.py --score_model umt-b16-25m-clip
# CUDA_VISIBLE_DEVICES=0 python cam_motion_pairwise_test_complex_caption_temp.py --score_model umt-b16-25m-itm
# CUDA_VISIBLE_DEVICES=2 python cam_motion_pairwise_test_complex_caption_temp.py --score_model umt-l16-25m-clip
# CUDA_VISIBLE_DEVICES=3 python cam_motion_pairwise_test_complex_caption_temp.py --score_model umt-l16-25m-itm
# CUDA_VISIBLE_DEVICES=4 python cam_motion_pairwise_test_complex_caption_temp.py --score_model languagebind-video-v1.5-ft
# CUDA_VISIBLE_DEVICES=5 python cam_motion_pairwise_test_complex_caption_temp.py --score_model languagebind-video-ft
# CUDA_VISIBLE_DEVICES=1 python cam_motion_pairwise_test_complex_caption_temp.py --score_model internvideo2-1b-stage2-clip
# CUDA_VISIBLE_DEVICES=7 python cam_motion_pairwise_test_complex_caption_temp.py --score_model internvideo2-1b-stage2-itm
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

from pairwise_caption_benchmark import (
    generate_pairwise_caption_dataset,
    PairwiseCaptionBenchmark,
    print_detailed_caption_task_statistics,
)
from pairwise_benchmark import (
    PairwiseBenchmark,
    # print_detailed_task_statistics,
    generate_pairwise_datasets,
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
parser.add_argument("--caption_save_dir", type=str, default="cam_motion-20250227_0324ground_only", help="Directory to save the sampled tasks")
parser.add_argument("--max_neg_samples", type=int, default=4, help="A maximum number of negative samples use per video")
parser.add_argument("--split", type=str, default="test", help="Split to use for the benchmark")
parser.add_argument("--collect_results", action="store_true", help="Whether to collect the results")
args = parser.parse_args()


print(f"Using score model: {args.score_model}")

score_model = t2v_metrics.get_score_model(model=args.score_model)

sampling_str = "top" if args.sampling == "top" else f"random_seed_{args.seed}"
folder_name = f"test_ratio_{1 - args.train_ratio:.2f}_num_{args.max_samples}_sampling_{sampling_str}"

question_str = args.question.replace(" ", "_")
SAVE_DIR = Path(args.save_dir) / folder_name / f"{args.score_model}_{question_str}"
assert SAVE_DIR.exists(), f"Directory {SAVE_DIR} does not exist, please run cam_motion_pairwise_test_temp.py first"

dataset = generate_pairwise_caption_dataset(
    caption_save_dir=args.caption_save_dir,
    max_samples=args.max_samples,
    max_neg_samples=args.max_neg_samples,
    split=args.split,
    video_root=VIDEO_ROOT,
    video_labels_dir=args.video_label_dir,
)
print_detailed_caption_task_statistics(
    dataset,
)

datasets = generate_pairwise_datasets(
    max_samples=args.max_samples,
    sampling=args.sampling,
    seed=args.seed,
    root=args.root,
    video_root=args.video_root,
    video_labels_dir=args.video_label_dir,
    train_ratio=args.train_ratio
)

# # Print statistics
# print_detailed_task_statistics(
#     datasets["original_tasks"]["raw"],
#     datasets["sampled_tasks"]["train"],
#     datasets["sampled_tasks"]["test"],
# )

sampled_tasks = datasets["sampled_tasks"]["test"]

if not args.collect_results:
    print(f"Running on Complex Description Task")
    # if type of score_model is VQAScoreModel
    if isinstance(score_model, t2v_metrics.VQAScore):
        benchmark = PairwiseCaptionBenchmark(dataset, mode="vqa")

        yes_kwargs = {"question_template": args.question, "answer_template": "Yes"}
        no_kwargs = {"question_template": args.question, "answer_template": "No"}

        yes_scores = score_model.batch_forward(benchmark, batch_size=args.batch_size, **yes_kwargs).cpu()
        no_scores = score_model.batch_forward(benchmark, batch_size=args.batch_size, **no_kwargs).cpu()

        # VQAScore Model can evaluate both retrieval and VQA
        retrieval_results, retrieval_tables = benchmark.evaluate_and_print_retrieval(yes_scores)
        vqa_results, vqa_tables = benchmark.evaluate_and_print_vqa(yes_scores, no_scores)

        # Save the results
        torch.save(retrieval_results, SAVE_DIR / "retrieval_results_complex_caption.pt")
        torch.save(vqa_results, SAVE_DIR / "vqa_results_complex_caption.pt")
        print(f"Saved results to {SAVE_DIR}/retrieval_results_complex_caption.pt")
        print(f"Saved results to {SAVE_DIR}/vqa_results_complex_caption.pt")

        # Save the tables in a text file
        with open(SAVE_DIR / "retrieval_tables_complex_caption.txt", "w") as f:
            f.write(retrieval_tables)
        with open(SAVE_DIR / "vqa_tables_complex_caption.txt", "w") as f:
            f.write(vqa_tables)
    else:
        benchmark = PairwiseCaptionBenchmark(dataset, mode="retrieval")
        scores = score_model.batch_forward(benchmark, batch_size=args.batch_size).cpu()

        # CLIPScore/ITMScore Model can only evaluate retrieval
        retrieval_results, retrieval_tables = benchmark.evaluate_and_print_retrieval(scores)

        # Save the results
        torch.save(retrieval_results, SAVE_DIR / "retrieval_results_complex_caption.pt")
        print(f"Saved results to {SAVE_DIR}/retrieval_results_complex_caption.pt")

        # Save the tables in a text file
        with open(SAVE_DIR / "retrieval_tables_complex_caption.txt", "w") as f:
            f.write(retrieval_tables)


# Define the skill mapping
SKILL_MAPPING = {
    "Motion & Steadiness": ["movement_and_steadiness"],
    "Scene Dynamics": ["scene_dynamics"],
    "Motion Speed": ["camera_movement_speed"],
    "Motion Direction": [
        "translation_direction",
        "rotation_direction",
        "object_centric_direction",
        "intrinsic_direction",
    ],
    "Confusable Motion": [
        "instrinsic_vs_extrinsic",
        "rotation_vs_translation",
        "reference_frame",
    ],
    "Has Motion": [
        "has_intrinsic_change",
        "has_translation",
        "has_rotation",
        "has_arc_crane",
    ],
    "Tracking Shot": ["special_tracking", "general_tracking"],
    "Only Motion": ["only_intrinsic_change", "only_translation", "only_rotation"],
}

CAPTION_SKILL_MAPPING = {
    "Complex Description": ["complex_description"],
}

RETRIEVAL_MAPPING = {
    "Skill-based Retrieval": ["Motion & Steadiness", "Scene Dynamics", "Motion Speed", "Motion Direction", "Confusable Motion", "Has Motion", "Tracking Shot", "Only Motion"],
    "Caption-based Retrieval": ["Complex Description"]
}


def count_samples_by_task(benchmark):
    """
    Creates a dictionary showing the number of samples per task using the benchmark object.

    Args:
        benchmark: The PairwiseBenchmark object that contains sample ID mappings

    Returns:
        Dictionary with the same structure as results, but with sample counts instead of performance metrics
    """
    # Start with the structure similar to results
    counts = {"overall": {"count": len(benchmark.samples)}, "skills": {}}

    total_samples = 0

    for skill in benchmark.skills:
        skill_sample_ids = benchmark.skill_to_sample_ids[skill]
        skill_sample_count = len(skill_sample_ids)

        counts["skills"][skill] = {
            "overall": {"count": skill_sample_count},
            "tasks": {},
        }

        for task in benchmark.skill_to_tasks[skill]:
            task_sample_ids = benchmark.task_to_sample_ids[task]
            task_sample_count = len(task_sample_ids)

            counts["skills"][skill]["tasks"][task] = {"count": task_sample_count}

        total_samples += skill_sample_count

    # Verify that all tasks sum up to the overall count
    assert total_samples == len(
        benchmark.samples
    ), f"Task sample sum ({total_samples}) doesn't match overall count ({len(benchmark.samples)})"

    return counts


def merge_and_rename_skills(results, skill_mapping, counts=None):
    """
    Merges and renames skills in the results dictionary according to the provided mapping.

    Args:
        results: Dictionary from evaluate_retrieval_scores or evaluate_vqa_scores
        skill_mapping: Dictionary mapping new skill names to lists of old skill names
        counts: Dictionary with sample counts

    Returns:
        New dictionary with merged and renamed skills
    """
    # Create a new results structure
    new_results = {"overall": results["overall"].copy(), "skills": {}}

    # Create reversed mapping (old_skill -> new_skill)
    old_to_new = {}
    for new_skill, old_skills in skill_mapping.items():
        for old_skill in old_skills:
            old_to_new[old_skill] = new_skill

    # Initialize new skills with empty structures
    for new_skill in skill_mapping:
        if "count" in results["overall"]:  # This is a count dictionary
            new_results["skills"][new_skill] = {"overall": {"count": 0}, "tasks": {}}
        else:  # This is a performance metrics dictionary
            # Copy the structure of the overall metrics for each new skill
            new_results["skills"][new_skill] = {
                "overall": {metric: 0.0 for metric in results["overall"]},
                "tasks": {},
            }

    # Collect all tasks and their data under the new skill groupings
    for old_skill, skill_data in results["skills"].items():
        if old_skill not in old_to_new:
            print(f"Warning: Skill '{old_skill}' not found in mapping, skipping")
            import pdb; pdb.set_trace()
            continue

        new_skill = old_to_new[old_skill]

        # Copy task data to the new skill structure
        for task, task_data in skill_data["tasks"].items():
            new_results["skills"][new_skill]["tasks"][task] = task_data.copy()

    # Calculate aggregated metrics for each new skill
    for new_skill, skill_data in new_results["skills"].items():
        if "count" in results["overall"]:  # This is a count dictionary
            # Sum up the counts
            skill_data["overall"]["count"] = sum(
                task_data["count"] for task_data in skill_data["tasks"].values()
            )
        else:  # This is a performance metrics dictionary
            # If we have counts, use them for weighted average
            if counts is not None:
                for metric in results["overall"]:
                    total_weighted_sum = 0.0
                    total_count = 0

                    for task, task_data in skill_data["tasks"].items():
                        if metric in task_data and task in counts["skills"].get(
                            new_skill, {}
                        ).get("tasks", {}):
                            task_count = counts["skills"][new_skill]["tasks"][task].get(
                                "count", 0
                            )
                            total_weighted_sum += task_data[metric] * task_count
                            total_count += task_count

                    if total_count > 0:
                        skill_data["overall"][metric] = total_weighted_sum / total_count
                    else:
                        import pdb; pdb.set_trace()
                        skill_data["overall"][metric] = 0.0
            else:
                # If no counts are provided, fall back to simple average
                for metric in results["overall"]:
                    tasks_with_metric = [
                        task_data[metric]
                        for task_data in skill_data["tasks"].values()
                        if metric in task_data
                    ]

                    if tasks_with_metric:
                        skill_data["overall"][metric] = sum(tasks_with_metric) / len(
                            tasks_with_metric
                        )
                    else:
                        skill_data["overall"][metric] = 0.0

    return new_results


def process_benchmark_results(results, benchmark, skill_mapping):
    """
    Process benchmark results by calculating sample counts and merging/renaming skills.

    Args:
        results: Dictionary from evaluate_retrieval_scores or evaluate_vqa_scores
        benchmark: The PairwiseBenchmark object
        skill_mapping: Dictionary mapping new skill names to lists of old skill names

    Returns:
        Tuple of (merged_results, merged_counts)
    """
    # First get the counts
    counts = count_samples_by_task(benchmark)

    # Then merge and rename skills in the counts
    merged_counts = merge_and_rename_skills(counts, skill_mapping)

    # Now merge and rename skills in the results, using the counts for weighted averaging
    merged_results = merge_and_rename_skills(
        results, skill_mapping, counts=merged_counts
    )

    return merged_results, merged_counts


def combine_results(
    results1, results2, counts1=None, counts2=None, family1_name=None, family2_name=None
):
    """
    Combines two sets of benchmark results into a single results dictionary,
    treating each set as a separate family of skills.

    Args:
        results1: First results dictionary
        results2: Second results dictionary
        counts1: Optional counts dictionary for the first results
        counts2: Optional counts dictionary for the second results
        family1_name: Optional prefix for skills from the first set (default: None)
        family2_name: Optional prefix for skills from the second set (default: None)

    Returns:
        Tuple of (combined_results, combined_counts) if counts are provided,
        otherwise just combined_results
    """
    # Check that both results have the same structure (same metrics)
    if set(results1["overall"].keys()) != set(results2["overall"].keys()):
        raise ValueError(
            "Results dictionaries have different metrics and cannot be combined"
        )

    # Create a new results dictionary with the same structure
    combined_results = {"overall": {}, "skills": {}}

    # Initialize combined counts if needed
    combined_counts = None
    if counts1 is not None and counts2 is not None:
        combined_counts = {"overall": {"count": 0}, "skills": {}}

    # Combine overall metrics (weighted average based on count if available)
    for metric in results1["overall"]:
        if metric == "count":
            # For count metrics, just sum them
            combined_results["overall"][metric] = (
                results1["overall"][metric] + results2["overall"][metric]
            )
            if combined_counts is not None:
                combined_counts["overall"][metric] = (
                    counts1["overall"][metric] + counts2["overall"][metric]
                )
        else:
            # For performance metrics, do weighted average if counts are available
            if counts1 is not None and counts2 is not None:
                weight1 = counts1["overall"]["count"]
                weight2 = counts2["overall"]["count"]
                total_weight = weight1 + weight2

                if total_weight > 0:
                    combined_results["overall"][metric] = (
                        (results1["overall"][metric] * weight1)
                        + (results2["overall"][metric] * weight2)
                    ) / total_weight
                else:
                    combined_results["overall"][metric] = 0.0
            else:
                # Simple average if no counts
                combined_results["overall"][metric] = (
                    results1["overall"][metric] + results2["overall"][metric]
                ) / 2

    # Process skills from results1
    for skill, skill_data in results1["skills"].items():
        # Add prefix to skill name if provided
        new_skill_name = f"{family1_name}: {skill}" if family1_name else skill

        # Copy skill data to combined results
        combined_results["skills"][new_skill_name] = {
            "overall": skill_data["overall"].copy(),
            "tasks": {
                task: task_data.copy()
                for task, task_data in skill_data["tasks"].items()
            },
        }

        # Copy count data if available
        if combined_counts is not None:
            combined_counts["skills"][new_skill_name] = {
                "overall": {"count": counts1["skills"][skill]["overall"]["count"]},
                "tasks": {
                    task: {"count": task_data["count"]}
                    for task, task_data in counts1["skills"][skill]["tasks"].items()
                },
            }

    # Process skills from results2
    for skill, skill_data in results2["skills"].items():
        # Add prefix to skill name if provided
        new_skill_name = f"{family2_name}: {skill}" if family2_name else skill

        # Copy skill data to combined results
        combined_results["skills"][new_skill_name] = {
            "overall": skill_data["overall"].copy(),
            "tasks": {
                task: task_data.copy()
                for task, task_data in skill_data["tasks"].items()
            },
        }

        # Copy count data if available
        if combined_counts is not None:
            combined_counts["skills"][new_skill_name] = {
                "overall": {"count": counts2["skills"][skill]["overall"]["count"]},
                "tasks": {
                    task: {"count": task_data["count"]}
                    for task, task_data in counts2["skills"][skill]["tasks"].items()
                },
            }

    # Return the combined results (and counts if provided)
    if combined_counts is not None:
        return combined_results, combined_counts
    else:
        return combined_results


if isinstance(score_model, t2v_metrics.VQAScore):
    # assert results are saved
    pairwise_benchmark = PairwiseBenchmark(sampled_tasks, mode="vqa")
    benchmark = PairwiseCaptionBenchmark(dataset, mode="vqa")
    pairwise_vqa_results = torch.load(SAVE_DIR / "vqa_results.pt")
    pairwise_vqa_results, pairwise_vqa_counts = process_benchmark_results(
        pairwise_vqa_results, pairwise_benchmark, SKILL_MAPPING
    )
    caption_vqa_results = torch.load(SAVE_DIR / "vqa_results_complex_caption.pt")
    caption_vqa_results, caption_vqa_counts = process_benchmark_results(
        caption_vqa_results, benchmark, CAPTION_SKILL_MAPPING
    )
    
    combined_results, combined_counts = combine_results(
        pairwise_vqa_results,
        caption_vqa_results,
        pairwise_vqa_counts,
        caption_vqa_counts
    )
    combined_vqa_results_str = benchmark.format_vqa_results(combined_results)
    print(combined_vqa_results_str)
    vqa_tables_for_paper = Path(SAVE_DIR / "vqa_tables_for_paper.txt")
    with open(vqa_tables_for_paper, "w") as f:
        f.write(combined_vqa_results_str)
    print(f"Saved VQA results (for paper) to {vqa_tables_for_paper}")
else:
    pairwise_benchmark = PairwiseBenchmark(sampled_tasks, mode="retrieval")
    benchmark = PairwiseCaptionBenchmark(dataset, mode="retrieval")
    
pairwise_retrieval_results = torch.load(SAVE_DIR / "retrieval_results.pt")
pairwise_retrieval_results, pairwise_retrieval_counts = process_benchmark_results(
    pairwise_retrieval_results, pairwise_benchmark, SKILL_MAPPING
)
pairwise_retrieval_results_str = pairwise_benchmark.format_retrieval_results(
    pairwise_retrieval_results
)
retrieval_tables_for_supp = Path(SAVE_DIR / "retrieval_tables_for_supp.txt")
with open(retrieval_tables_for_supp, "w") as f:
    f.write(pairwise_retrieval_results_str)
print(f"Saved retrieval results (for supp) to {retrieval_tables_for_supp}")


caption_retrieval_results = torch.load(SAVE_DIR / "retrieval_results_complex_caption.pt")
caption_retrieval_results, caption_retrieval_counts = process_benchmark_results(
    caption_retrieval_results, benchmark, CAPTION_SKILL_MAPPING
)

combine_results, combined_counts = combine_results(
    pairwise_retrieval_results,
    caption_retrieval_results,
    pairwise_retrieval_counts,
    caption_retrieval_counts
)
combine_results, combined_counts = process_benchmark_results(
    combine_results, benchmark, RETRIEVAL_MAPPING
)
combined_retrieval_results_str = benchmark.format_retrieval_results(combine_results)
print(combined_retrieval_results_str)
retrieval_tables_for_paper = Path(SAVE_DIR / "retrieval_tables_for_paper.txt")
with open(retrieval_tables_for_paper, "w") as f:
    f.write(combined_retrieval_results_str)
