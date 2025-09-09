#!/usr/bin/env python3
"""
Method-agnostic binary classification evaluator.
Takes standardized score files and computes evaluation metrics (mAP, etc.).
This script works with any method that outputs scores in the expected format.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

def load_score_file(score_file):
    """Load scores from a standardized score file"""
    with open(score_file, 'r') as f:
        data = json.load(f)
    
    return data

def extract_scores_and_labels(score_data):
    """Extract scores and labels from score data"""
    scores = []
    labels = []
    
    for result in score_data["scores"]:
        if result["error"] is None:  # Only include successful samples
            scores.append(result["score"])
            # Convert ground truth label to binary
            label = 1 if result["ground_truth_label"].lower() == 'yes' else 0
            labels.append(label)
    
    return np.array(scores), np.array(labels)

def compute_binary_classification_metrics(scores, labels):
    """Compute comprehensive binary classification metrics"""
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels")
        return {
            "average_precision": 0.0,
            "roc_auc": 0.0,
            "num_samples": len(scores),
            "num_positive": np.sum(labels == 1),
            "num_negative": np.sum(labels == 0)
        }
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Compute metrics
    average_precision = average_precision_score(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    
    # Additional statistics
    num_samples = len(scores)
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    
    metrics = {
        "average_precision": float(average_precision),
        "roc_auc": float(roc_auc),
        "num_samples": int(num_samples),
        "num_positive": int(num_positive),
        "num_negative": int(num_negative),
        "positive_ratio": float(num_positive / num_samples) if num_samples > 0 else 0.0
    }
    
    return metrics

def generate_plots(scores, labels, output_dir, method_name, split_name):
    """Generate precision-recall and ROC curves"""
    if len(np.unique(labels)) < 2:
        print("Cannot generate plots: only one class present")
        return
    
    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    
    plt.figure(figsize=(10, 4))
    
    # PR Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\n{method_name} - {split_name}')
    plt.grid(True, alpha=0.3)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, 'r-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve\n{method_name} - {split_name}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{method_name}_{split_name}_curves.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {plot_path}")

def evaluate_single_file(score_file, generate_plots_flag=False, output_dir=None, quiet=False):
    """Evaluate a single score file"""
    if not quiet:
        print(f"\nEvaluating: {score_file}")
    
    # Load score data
    score_data = load_score_file(score_file)
    
    # Extract metadata
    metadata = score_data.get("metadata", {})
    method_name = metadata.get("method_type", "Unknown_Method")
    model_name = metadata.get("model_name", "Unknown_Model")
    checkpoint = metadata.get("checkpoint", "")
    split_name = metadata.get("split_name", Path(score_file).stem)
    
    # Create a unique identifier that includes model and checkpoint info
    if checkpoint:
        clean_checkpoint = checkpoint.split("/")[-1] if "/" in checkpoint else checkpoint
        unique_id = f"{model_name}_{clean_checkpoint}_{split_name}"
    else:
        unique_id = f"{model_name}_{split_name}"
    
    if not quiet:
        print(f"Model: {model_name}")
        if checkpoint:
            print(f"Checkpoint: {checkpoint}")
        print(f"Split: {split_name}")
    
    # Calculate statistics from the scores data
    total_samples = len(score_data["scores"])
    failed_samples = sum(1 for result in score_data["scores"] if result["error"] is not None)
    successful_samples = total_samples - failed_samples
    
    if not quiet:
        print(f"Total samples in file: {total_samples}")
        print(f"Successful samples: {successful_samples}")
        print(f"Failed samples: {failed_samples}")
    
    # Extract scores and labels
    scores, labels = extract_scores_and_labels(score_data)
    
    if len(scores) == 0:
        if not quiet:
            print("No valid scores found in file")
        return unique_id, None
    
    # Compute metrics
    metrics = compute_binary_classification_metrics(scores, labels)
    
    # Print results
    if not quiet:
        print(f"Evaluation Results:")
        print(f"  Average Precision (mAP): {metrics['average_precision']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Valid samples used: {metrics['num_samples']}")
        print(f"  Positive samples: {metrics['num_positive']}")
        print(f"  Negative samples: {metrics['num_negative']}")
        print(f"  Positive ratio: {metrics['positive_ratio']:.3f}")
    
    # Generate plots if requested
    if generate_plots_flag and output_dir:
        # Use unique_id for plot naming to avoid conflicts
        generate_plots(scores, labels, output_dir, unique_id.replace("_", "-"), split_name)
    
    # Add metadata to metrics
    metrics["metadata"] = metadata
    metrics["split_name"] = split_name
    metrics["model_name"] = model_name
    metrics["checkpoint"] = checkpoint
    metrics["unique_id"] = unique_id
    
    return unique_id, metrics

def save_evaluation_results(results, output_file):
    """Save evaluation results to JSON file"""
    # Compute overall statistics
    valid_maps = [r["average_precision"] for r in results.values() if r is not None]
    valid_aucs = [r["roc_auc"] for r in results.values() if r is not None]
    
    overall_stats = None
    if valid_maps:
        overall_stats = {
            "mean_average_precision": float(np.mean(valid_maps)),
            "std_average_precision": float(np.std(valid_maps)) if len(valid_maps) > 1 else 0.0,
            "mean_roc_auc": float(np.mean(valid_aucs)),
            "std_roc_auc": float(np.std(valid_aucs)) if len(valid_aucs) > 1 else 0.0,
            "evaluated_splits": len(valid_maps)
        }
    
    # Prepare summary with overall stats at the top level
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "overall_average_precision": overall_stats["mean_average_precision"] if overall_stats else None,
        "overall_roc_auc": overall_stats["mean_roc_auc"] if overall_stats else None,
        "total_splits": len(results),
        "evaluated_splits": len(valid_maps),
        "overall_statistics": overall_stats,
        "results_by_split": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_file}")
    if overall_stats:
        print(f"Overall Average Precision: {overall_stats['mean_average_precision']:.4f}")
        print(f"Overall ROC AUC: {overall_stats['mean_roc_auc']:.4f}")

def print_hierarchical_results(results):
    """Print results organized hierarchically by model and checkpoint"""
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to display")
        return
    
    # Group results by model and checkpoint
    grouped_results = defaultdict(lambda: defaultdict(list))
    
    for split_name, metrics in valid_results.items():
        model_name = metrics.get("model_name", "Unknown_Model")
        checkpoint = metrics.get("checkpoint", "")
        
        # Create a model key that includes checkpoint info
        if checkpoint:
            model_key = (model_name, checkpoint)
        else:
            model_key = (model_name, "")
        
        grouped_results[model_key]["splits"].append((split_name, metrics))
    
    # Print results hierarchically
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    
    for (model_name, checkpoint), group_data in grouped_results.items():
        # Print model header
        print(f"\n┏━━ MODEL: {model_name}")
        if checkpoint:
            print(f"┃   Checkpoint: {checkpoint}")
        print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Print splits for this model
        splits = group_data["splits"]
        for i, (split_name, metrics) in enumerate(splits):
            split_display_name = metrics.get("split_name", split_name.split("_")[-1])
            
            # Use different symbols for different splits
            if i == len(splits) - 1:  # Last split
                prefix = "   └─"
            else:
                prefix = "   ├─"
            
            print(f"{prefix} {split_display_name:25s}: mAP = {metrics['average_precision']:.4f}, AUC = {metrics['roc_auc']:.4f} "
                  f"({metrics['num_samples']} samples, {metrics['num_positive']}+/{metrics['num_negative']}-)")
        
        # If multiple splits for this model, show model average
        if len(splits) > 1:
            model_maps = [metrics["average_precision"] for _, metrics in splits]
            model_aucs = [metrics["roc_auc"] for _, metrics in splits]
            total_samples = sum(metrics["num_samples"] for _, metrics in splits)
            
            print(f"   {'─' * 25} Model Average: mAP = {np.mean(model_maps):.4f}, AUC = {np.mean(model_aucs):.4f} "
                  f"({total_samples} total samples)")

def find_score_files(score_dir):
    """Find all binary classification score files in a directory"""
    score_dir = Path(score_dir)
    if not score_dir.exists():
        return []
    
    # Look for binary classification score files with the simplified pattern
    score_files = list(score_dir.glob("classification_scores_*.json"))
    
    # Remove duplicates and sort
    score_files = sorted(list(set(score_files)))
    return score_files

def main():
    parser = argparse.ArgumentParser(description='Method-agnostic binary classification evaluator')
    parser.add_argument('score_files', nargs='*', default=[],
                      help='Score files to evaluate (JSON format from score generation step). If not provided, auto-discovers files in --score_dir')
    parser.add_argument('--score_dir', type=str, default='scores',
                      help='Directory to search for score files (used when no specific files provided)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--plots', action='store_true',
                      help='Generate precision-recall and ROC curves')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file name for results (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Determine which score files to process
    if args.score_files:
        # Use explicitly provided files
        score_files = [Path(f) for f in args.score_files]
        print(f"Using {len(score_files)} explicitly provided score files")
    else:
        # Auto-discover score files in the score directory
        score_files = find_score_files(args.score_dir)
        if not score_files:
            print(f"No binary classification score files found in {args.score_dir}")
            print("Looking for files matching: vqa_scores_*.json")
            return
        print(f"Auto-discovered {len(score_files)} score files in {args.score_dir}")
        for f in score_files:
            print(f"  - {f.name}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("METHOD-AGNOSTIC BINARY CLASSIFICATION EVALUATION")
    print(f"{'='*80}")
    print(f"Score files to evaluate: {len(score_files)}")
    
    # Evaluate each file (quietly to avoid cluttering output)
    results = {}
    
    print("\nProcessing files...")
    for i, score_file in enumerate(score_files, 1):
        if not score_file.exists():
            print(f"Warning: Score file not found: {score_file}")
            continue
        
        print(f"  [{i}/{len(score_files)}] {score_file.name}")
        
        split_name, metrics = evaluate_single_file(
            score_file, 
            generate_plots_flag=args.plots,
            output_dir=output_dir,
            quiet=True  # Suppress detailed per-file output
        )
        
        results[split_name] = metrics
    
    # Print hierarchical results
    print_hierarchical_results(results)
    
    # Print overall statistics
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 1:
        maps = [m["average_precision"] for m in valid_results.values()]
        aucs = [m["roc_auc"] for m in valid_results.values()]
        
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print(f"{'='*80}")
        if len(valid_results) > 1:
            print(f"Overall Average (across all splits): mAP = {np.mean(maps):.4f} (±{np.std(maps):.4f}), AUC = {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")
        else:
            print(f"Single Split Result: mAP = {np.mean(maps):.4f}, AUC = {np.mean(aucs):.4f}")
        print(f"Total splits evaluated: {len(valid_results)}")
    
    # Generate output filename if not provided
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # Create a descriptive filename without timestamp
        # Include number of models/files for clarity
        num_models = len(set([m.get("model_name", "unknown") for m in valid_results.values() if m and "metadata" in m]))
        num_files = len(valid_results)
        output_file = output_dir / f"binary_classification_evaluation_{num_models}models_{num_files}files.json"
    
    # Save results
    save_evaluation_results(valid_results, output_file)
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()