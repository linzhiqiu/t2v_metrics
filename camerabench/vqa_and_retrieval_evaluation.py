#!/usr/bin/env python3
"""
Method-agnostic VQA and Retrieval evaluator.
Takes standardized score files and computes evaluation metrics for both VQA and retrieval tasks.
This script works with any method that outputs scores in the expected format.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

def load_score_file(score_file):
    """Load scores from a standardized score file"""
    with open(score_file, 'r') as f:
        data = json.load(f)
    
    return data

def extract_score_matrices(score_data):
    """Extract yes and no score matrices from score data"""
    yes_scores = []
    no_scores = []
    
    for result in score_data["scores"]:
        if result["error"] is None:  # Only include successful samples
            # Convert dictionary format to matrix format for compatibility
            yes_result = [
                result["yes_scores"]["pos_text_pos_image"],
                result["yes_scores"]["pos_text_neg_image"],
                result["yes_scores"]["neg_text_pos_image"],
                result["yes_scores"]["neg_text_neg_image"]
            ]
            no_result = [
                result["no_scores"]["pos_text_pos_image"],
                result["no_scores"]["pos_text_neg_image"],
                result["no_scores"]["neg_text_pos_image"],
                result["no_scores"]["neg_text_neg_image"]
            ]
            
            yes_scores.append(yes_result)
            no_scores.append(no_result)
    
    return np.array(yes_scores), np.array(no_scores)

def compute_retrieval_scores_from_vqa(yes_scores):
    """Convert VQA yes scores to retrieval format (using only yes scores like original)"""
    retrieval_scores = []
    
    for yes_result in yes_scores:
        # Convert to retrieval dictionary format using yes scores
        # yes_result = [pos_text_pos_image, pos_text_neg_image, neg_text_pos_image, neg_text_neg_image]
        score_dict = {
            "pos_text_pos_image": yes_result[0],
            "pos_text_neg_image": yes_result[1], 
            "neg_text_pos_image": yes_result[2],
            "neg_text_neg_image": yes_result[3]
        }
        retrieval_scores.append(score_dict)
    
    return retrieval_scores

def evaluate_vqa_metrics(yes_scores, no_scores):
    """Evaluate VQA metrics - binary_acc and question_acc matching original implementation"""
    if len(yes_scores) == 0:
        return {"binary_acc": 0.0, "question_acc": 0.0, "num_samples": 0}
    
    binary_correct = 0
    question_correct = 0
    total_binary = 0
    total_questions = 0
    
    for yes_result, no_result in zip(yes_scores, no_scores):
        # Convert to dictionary format for easier parsing
        yes_dict = {
            "pos_text_pos_image": yes_result[0],
            "pos_text_neg_image": yes_result[1], 
            "neg_text_pos_image": yes_result[2],
            "neg_text_neg_image": yes_result[3]
        }
        no_dict = {
            "pos_text_pos_image": no_result[0],
            "pos_text_neg_image": no_result[1], 
            "neg_text_pos_image": no_result[2],
            "neg_text_neg_image": no_result[3]
        }
        
        # Binary accuracy: each of 4 combinations should be correct
        # Positive text + positive image: yes > no
        binary_correct += 1 if yes_dict["pos_text_pos_image"] > no_dict["pos_text_pos_image"] else 0
        # Positive text + negative image: no > yes  
        binary_correct += 1 if no_dict["pos_text_neg_image"] > yes_dict["pos_text_neg_image"] else 0
        # Negative text + positive image: no > yes
        binary_correct += 1 if no_dict["neg_text_pos_image"] > yes_dict["neg_text_pos_image"] else 0
        # Negative text + negative image: yes > no
        binary_correct += 1 if yes_dict["neg_text_neg_image"] > no_dict["neg_text_neg_image"] else 0
        total_binary += 4
        
        # Question accuracy: both images correct for each question (original implementation)
        # Positive question correct: both pos_text+pos_image and pos_text+neg_image correct
        pos_question_correct = (yes_dict["pos_text_pos_image"] > no_dict["pos_text_pos_image"]) and \
                              (no_dict["pos_text_neg_image"] > yes_dict["pos_text_neg_image"])
        # Negative question correct: both neg_text+pos_image and neg_text+neg_image correct  
        neg_question_correct = (no_dict["neg_text_pos_image"] > yes_dict["neg_text_pos_image"]) and \
                              (yes_dict["neg_text_neg_image"] > no_dict["neg_text_neg_image"])
        
        question_correct += (1 if pos_question_correct else 0) + (1 if neg_question_correct else 0)
        total_questions += 2
    
    return {
        "binary_acc": binary_correct / total_binary if total_binary > 0 else 0.0,
        "question_acc": question_correct / total_questions if total_questions > 0 else 0.0,
        "num_samples": len(yes_scores)
    }

def evaluate_retrieval_metrics(scores):
    """Evaluate retrieval metrics - text, image, group"""
    if len(scores) == 0:
        return {"text": 0.0, "image": 0.0, "group": 0.0, "num_samples": 0}
    
    text_correct = 0
    image_correct = 0  
    group_correct = 0
    
    def text_correct_check(result):
        """Check if text retrieval is correct for this sample."""
        return (result["pos_text_pos_image"] > result["neg_text_pos_image"] and 
                result["neg_text_neg_image"] > result["pos_text_neg_image"])
    
    def image_correct_check(result):
        """Check if image retrieval is correct for this sample."""
        return (result["pos_text_pos_image"] > result["pos_text_neg_image"] and 
                result["neg_text_neg_image"] > result["neg_text_pos_image"])
    
    def group_correct_check(result):
        """Check if both text and image retrieval are correct."""
        return image_correct_check(result) and text_correct_check(result)
    
    for score_dict in scores:
        text_correct += 1 if text_correct_check(score_dict) else 0
        image_correct += 1 if image_correct_check(score_dict) else 0
        group_correct += 1 if group_correct_check(score_dict) else 0
    
    total = len(scores)
    return {
        "text": text_correct / total,
        "image": image_correct / total, 
        "group": group_correct / total,
        "num_samples": total
    }

def evaluate_single_file(score_file, mode='both', quiet=False):
    """Evaluate a single score file for VQA and/or retrieval metrics"""
    if not quiet:
        print(f"\nEvaluating: {score_file}")
    
    # Load score data
    score_data = load_score_file(score_file)
    
    # Extract metadata
    metadata = score_data.get("metadata", {})
    method_name = metadata.get("method_type", "Unknown_Method")
    model_name = metadata.get("model_name", "Unknown_Model")
    checkpoint = metadata.get("checkpoint", "")
    skill_name = metadata.get("skill_name", "Unknown_Skill")
    task_name = metadata.get("task_name", "")
    split_name = metadata.get("split_name", skill_name)
    
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
        print(f"Method: {method_name}")
        print(f"Skill: {skill_name}")
        if task_name:
            print(f"Task: {task_name}")
        print(f"Split: {split_name}")
    
    # Calculate statistics from the scores data
    total_samples = len(score_data["scores"])
    failed_samples = sum(1 for result in score_data["scores"] if result["error"] is not None)
    successful_samples = total_samples - failed_samples
    
    if not quiet:
        print(f"Total samples in file: {total_samples}")
        print(f"Successful samples: {successful_samples}")
        print(f"Failed samples: {failed_samples}")
    
    # Extract score matrices
    yes_scores, no_scores = extract_score_matrices(score_data)
    
    if len(yes_scores) == 0:
        if not quiet:
            print("No valid scores found in file")
        return unique_id, None
    
    results = {
        "split_name": split_name,
        "skill_name": skill_name, 
        "task_name": task_name, 
        "model_name": model_name,
        "checkpoint": checkpoint,
        "unique_id": unique_id,
        "metadata": metadata
    }
    
    # Evaluate VQA metrics
    if mode in ['vqa', 'both']:
        vqa_metrics = evaluate_vqa_metrics(yes_scores, no_scores)
        results["vqa"] = vqa_metrics
        
        if not quiet:
            print(f"VQA Results:")
            print(f"  Binary Accuracy: {vqa_metrics['binary_acc']:.4f}")
            print(f"  Question Accuracy: {vqa_metrics['question_acc']:.4f}")
            print(f"  Valid samples used: {vqa_metrics['num_samples']}")
    
    # Evaluate Retrieval metrics
    if mode in ['retrieval', 'both']:
        retrieval_scores = compute_retrieval_scores_from_vqa(yes_scores)
        retrieval_metrics = evaluate_retrieval_metrics(retrieval_scores)
        results["retrieval"] = retrieval_metrics
        
        if not quiet:
            print(f"Retrieval Results:")
            print(f"  Text: {retrieval_metrics['text']:.4f}")
            print(f"  Image: {retrieval_metrics['image']:.4f}")
            print(f"  Group: {retrieval_metrics['group']:.4f}")
            print(f"  Valid samples used: {retrieval_metrics['num_samples']}")
    
    return unique_id, results

def print_hierarchical_results(results, mode):
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
            
            # Show retrieval first, then VQA (as requested)
            result_parts = []
            if mode in ['retrieval', 'both'] and 'retrieval' in metrics:
                ret = metrics['retrieval']
                result_parts.append(f"Text = {ret['text']:.4f}, Image = {ret['image']:.4f}, Group = {ret['group']:.4f}")
            
            if mode in ['vqa', 'both'] and 'vqa' in metrics:
                vqa = metrics['vqa']
                result_parts.append(f"Binary = {vqa['binary_acc']:.4f}, Question = {vqa['question_acc']:.4f}")
            
            result_str = " | ".join(result_parts)
            
            print(f"{prefix} {split_display_name:25s}: {result_str} "
                  f"({metrics.get('vqa', metrics.get('retrieval', {})).get('num_samples', 0)} samples)")
        
        # If multiple splits for this model, show model average
        if len(splits) > 1:
            model_averages = []
            total_samples = 0
            
            if mode in ['retrieval', 'both']:
                retrieval_splits = [metrics for _, metrics in splits if 'retrieval' in metrics]
                if retrieval_splits:
                    text_scores = [m['retrieval']['text'] for m in retrieval_splits]
                    image_scores = [m['retrieval']['image'] for m in retrieval_splits]
                    group_scores = [m['retrieval']['group'] for m in retrieval_splits]
                    model_averages.append(f"Text = {np.mean(text_scores):.4f}, Image = {np.mean(image_scores):.4f}, Group = {np.mean(group_scores):.4f}")
                    total_samples = sum(m['retrieval']['num_samples'] for m in retrieval_splits)
            
            if mode in ['vqa', 'both']:
                vqa_splits = [metrics for _, metrics in splits if 'vqa' in metrics]
                if vqa_splits:
                    binary_scores = [m['vqa']['binary_acc'] for m in vqa_splits]
                    question_scores = [m['vqa']['question_acc'] for m in vqa_splits]
                    model_averages.append(f"Binary = {np.mean(binary_scores):.4f}, Question = {np.mean(question_scores):.4f}")
                    if total_samples == 0:  # Only set if not already set by retrieval
                        total_samples = sum(m['vqa']['num_samples'] for m in vqa_splits)
            
            avg_str = " | ".join(model_averages)
            print(f"   {'─' * 25} Model Average: {avg_str} ({total_samples} total samples)")

def save_evaluation_results(results, mode, output_file):
    """Save evaluation results to JSON file"""
    # Compute overall statistics
    overall_stats = {}
    
    if mode in ['vqa', 'both']:
        valid_vqa = [r["vqa"] for r in results.values() if r is not None and "vqa" in r]
        if valid_vqa:
            binary_accs = [v["binary_acc"] for v in valid_vqa]
            question_accs = [v["question_acc"] for v in valid_vqa]
            
            overall_stats["vqa"] = {
                "mean_binary_acc": float(np.mean(binary_accs)),
                "std_binary_acc": float(np.std(binary_accs)) if len(binary_accs) > 1 else 0.0,
                "mean_question_acc": float(np.mean(question_accs)),
                "std_question_acc": float(np.std(question_accs)) if len(question_accs) > 1 else 0.0,
                "evaluated_splits": len(valid_vqa)
            }
    
    if mode in ['retrieval', 'both']:
        valid_retrieval = [r["retrieval"] for r in results.values() if r is not None and "retrieval" in r]
        if valid_retrieval:
            # Overall retrieval averages
            text_scores = [v["text"] for v in valid_retrieval]
            image_scores = [v["image"] for v in valid_retrieval]
            group_scores = [v["group"] for v in valid_retrieval]
            
            overall_stats["retrieval"] = {
                "mean_text": float(np.mean(text_scores)),
                "std_text": float(np.std(text_scores)) if len(text_scores) > 1 else 0.0,
                "mean_image": float(np.mean(image_scores)),
                "std_image": float(np.std(image_scores)) if len(image_scores) > 1 else 0.0,
                "mean_group": float(np.mean(group_scores)),
                "std_group": float(np.std(group_scores)) if len(group_scores) > 1 else 0.0,
                "evaluated_splits": len(valid_retrieval)
            }
            
            # Separate skill-based vs caption-based averages
            skill_retrieval = []
            caption_retrieval = []
            
            for split_name, metrics in results.items():
                if metrics and "retrieval" in metrics:
                    skill_name = metrics.get("skill_name", "")
                    if "complex description" in skill_name.lower() or "caption" in skill_name.lower():
                        caption_retrieval.append(metrics["retrieval"])
                    else:
                        skill_retrieval.append(metrics["retrieval"])
            
            if skill_retrieval:
                skill_text = [v["text"] for v in skill_retrieval]
                skill_image = [v["image"] for v in skill_retrieval]
                skill_group = [v["group"] for v in skill_retrieval]
                
                overall_stats["retrieval_skill_based"] = {
                    "mean_text": float(np.mean(skill_text)),
                    "std_text": float(np.std(skill_text)) if len(skill_text) > 1 else 0.0,
                    "mean_image": float(np.mean(skill_image)),
                    "std_image": float(np.std(skill_image)) if len(skill_image) > 1 else 0.0,
                    "mean_group": float(np.mean(skill_group)),
                    "std_group": float(np.std(skill_group)) if len(skill_group) > 1 else 0.0,
                    "evaluated_splits": len(skill_retrieval)
                }
            
            if caption_retrieval:
                caption_text = [v["text"] for v in caption_retrieval]
                caption_image = [v["image"] for v in caption_retrieval]
                caption_group = [v["group"] for v in caption_retrieval]
                
                overall_stats["retrieval_caption_based"] = {
                    "mean_text": float(np.mean(caption_text)),
                    "std_text": float(np.std(caption_text)) if len(caption_text) > 1 else 0.0,
                    "mean_image": float(np.mean(caption_image)),
                    "std_image": float(np.std(caption_image)) if len(caption_image) > 1 else 0.0,
                    "mean_group": float(np.mean(caption_group)),
                    "std_group": float(np.std(caption_group)) if len(caption_group) > 1 else 0.0,
                    "evaluated_splits": len(caption_retrieval)
                }
    
    # Prepare summary with top-level overall averages for easy access (matching binary classification format)
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "evaluation_mode": mode,
        "total_splits": len(results),
        "evaluated_splits": len([r for r in results.values() if r is not None])
    }
    
    # Add top-level overall averages for easy access (matching binary classification format)
    if "vqa" in overall_stats:
        summary["overall_binary_acc"] = overall_stats["vqa"]["mean_binary_acc"]
        summary["overall_question_acc"] = overall_stats["vqa"]["mean_question_acc"]
    
    if "retrieval" in overall_stats:
        summary["overall_retrieval_text"] = overall_stats["retrieval"]["mean_text"]
        summary["overall_retrieval_image"] = overall_stats["retrieval"]["mean_image"]
        summary["overall_retrieval_group"] = overall_stats["retrieval"]["mean_group"]
    
    if "retrieval_skill_based" in overall_stats:
        summary["skill_based_retrieval_text"] = overall_stats["retrieval_skill_based"]["mean_text"]
        summary["skill_based_retrieval_image"] = overall_stats["retrieval_skill_based"]["mean_image"]
        summary["skill_based_retrieval_group"] = overall_stats["retrieval_skill_based"]["mean_group"]
    
    if "retrieval_caption_based" in overall_stats:
        summary["caption_based_retrieval_text"] = overall_stats["retrieval_caption_based"]["mean_text"]
        summary["caption_based_retrieval_image"] = overall_stats["retrieval_caption_based"]["mean_image"]
        summary["caption_based_retrieval_group"] = overall_stats["retrieval_caption_based"]["mean_group"]
    
    # Add detailed statistics (matching binary classification format)
    summary["overall_statistics"] = overall_stats
    summary["results_by_split"] = results
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_file}")
    
    # Print overall averages when saving
    if "retrieval" in overall_stats:
        ret_stats = overall_stats["retrieval"]
        print(f"Overall Retrieval Text: {ret_stats['mean_text']:.4f}")
        print(f"Overall Retrieval Image: {ret_stats['mean_image']:.4f}")
        print(f"Overall Retrieval Group: {ret_stats['mean_group']:.4f}")
    
    if "vqa" in overall_stats:
        vqa_stats = overall_stats["vqa"]
        print(f"Overall VQA Binary Accuracy: {vqa_stats['mean_binary_acc']:.4f}")
        print(f"Overall VQA Question Accuracy: {vqa_stats['mean_question_acc']:.4f}")
    
    if "retrieval_skill_based" in overall_stats:
        skill_stats = overall_stats["retrieval_skill_based"]
        print(f"Skill-based Retrieval Text: {skill_stats['mean_text']:.4f}")
        print(f"Skill-based Retrieval Image: {skill_stats['mean_image']:.4f}")
        print(f"Skill-based Retrieval Group: {skill_stats['mean_group']:.4f}")
    
    if "retrieval_caption_based" in overall_stats:
        caption_stats = overall_stats["retrieval_caption_based"]
        print(f"Caption-based Retrieval Text: {caption_stats['mean_text']:.4f}")
        print(f"Caption-based Retrieval Image: {caption_stats['mean_image']:.4f}")
        print(f"Caption-based Retrieval Group: {caption_stats['mean_group']:.4f}")

def find_score_files(score_dir):
    """Find all VQA/retrieval score files in a directory"""
    score_dir = Path(score_dir)
    if not score_dir.exists():
        return []
    
    # Look for VQA/retrieval score files with the simplified pattern
    score_files = list(score_dir.glob("vqa_retrieval_scores_*.json"))
    
    # Remove duplicates and sort
    score_files = sorted(list(set(score_files)))
    return score_files

def main():
    parser = argparse.ArgumentParser(description='Method-agnostic VQA and Retrieval evaluator')
    parser.add_argument('score_files', nargs='*', default=[],
                      help='Score files to evaluate (JSON format from score generation step). If not provided, auto-discovers files in --score_dir')
    parser.add_argument('--score_dir', type=str, default='scores',
                      help='Directory to search for score files (used when no specific files provided)')
    parser.add_argument('--mode', type=str, choices=['vqa', 'retrieval', 'both'], default='both',
                      help='Evaluation mode')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
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
            print(f"No VQA/retrieval score files found in {args.score_dir}")
            print("Looking for files matching: vqa_retrieval_scores_*.json")
            return
        print(f"Auto-discovered {len(score_files)} score files in {args.score_dir}")
        for f in score_files:
            print(f"  - {f.name}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("METHOD-AGNOSTIC VQA & RETRIEVAL EVALUATION")
    print(f"{'='*80}")
    print(f"Score files to evaluate: {len(score_files)}")
    print(f"Evaluation mode: {args.mode}")
    
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
            args.mode,
            quiet=True  # Suppress detailed per-file output
        )
        
        if metrics:
            results[split_name] = metrics
    
    # Print hierarchical results
    print_hierarchical_results(results, args.mode)
    
    # Print overall statistics
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 1:
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print(f"{'='*80}")
        
        overall_parts = []
        
        if args.mode in ['retrieval', 'both']:
            retrieval_metrics = [m["retrieval"] for m in valid_results.values() if "retrieval" in m]
            if retrieval_metrics:
                text_scores = [v["text"] for v in retrieval_metrics]
                image_scores = [v["image"] for v in retrieval_metrics]
                group_scores = [v["group"] for v in retrieval_metrics]
                if len(retrieval_metrics) > 1:
                    overall_parts.append(f"Text = {np.mean(text_scores):.4f} (±{np.std(text_scores):.4f}), Image = {np.mean(image_scores):.4f} (±{np.std(image_scores):.4f}), Group = {np.mean(group_scores):.4f} (±{np.std(group_scores):.4f})")
                else:
                    overall_parts.append(f"Text = {np.mean(text_scores):.4f}, Image = {np.mean(image_scores):.4f}, Group = {np.mean(group_scores):.4f}")
        
        if args.mode in ['vqa', 'both']:
            vqa_metrics = [m["vqa"] for m in valid_results.values() if "vqa" in m]
            if vqa_metrics:
                binary_accs = [v["binary_acc"] for v in vqa_metrics]
                question_accs = [v["question_acc"] for v in vqa_metrics]
                if len(vqa_metrics) > 1:
                    overall_parts.append(f"VQA: Binary = {np.mean(binary_accs):.4f} (±{np.std(binary_accs):.4f}), Question = {np.mean(question_accs):.4f} (±{np.std(question_accs):.4f})")
                else:
                    overall_parts.append(f"VQA: Binary = {np.mean(binary_accs):.4f}, Question = {np.mean(question_accs):.4f}")
        
        if overall_parts:
            result_label = "Overall Average (across all splits)" if len(valid_results) > 1 else "Single Split Result"
            overall_str = " | ".join(overall_parts)
            print(f"{result_label}: {overall_str}")
        
        print(f"Total splits evaluated: {len(valid_results)}")
    
    # Generate output filename if not provided
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # Create a descriptive filename without timestamp
        # Include number of models/files for clarity
        num_models = len(set([m.get("model_name", "unknown") for m in valid_results.values() if m and "metadata" in m]))
        num_files = len(valid_results)
        output_file = output_dir / f"vqa_retrieval_evaluation_{args.mode}_{num_models}models_{num_files}files.json"
    
    # Save results
    save_evaluation_results(valid_results, args.mode, output_file)
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()