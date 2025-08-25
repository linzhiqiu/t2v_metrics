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
        return {"binary_acc": 0.0, "question_acc": 0.0}
    
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
        "question_acc": question_correct / total_questions if total_questions > 0 else 0.0
    }

def evaluate_retrieval_metrics(scores):
    """Evaluate retrieval metrics - text, image, group"""
    if len(scores) == 0:
        return {"text": 0.0, "image": 0.0, "group": 0.0}
    
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
        "group": group_correct / total
    }

def evaluate_single_file(score_file, mode='both'):
    """Evaluate a single score file for VQA and/or retrieval metrics"""
    print(f"\nEvaluating: {score_file}")
    
    # Load score data
    score_data = load_score_file(score_file)
    
    # Extract metadata
    metadata = score_data.get("metadata", {})
    method_name = metadata.get("method_type", "Unknown_Method")
    skill_name = metadata.get("skill_name", "Unknown_Skill")
    task_name = metadata.get("task_name", "")
    
    print(f"Method: {method_name}")
    print(f"Skill: {skill_name}")
    if task_name:
        print(f"Task: {task_name}")
    print(f"Total samples in file: {score_data['total_samples']}")
    print(f"Successful samples: {score_data['successful_samples']}")
    print(f"Failed samples: {score_data['failed_samples']}")
    
    # Extract score matrices
    yes_scores, no_scores = extract_score_matrices(score_data)
    
    if len(yes_scores) == 0:
        print("No valid scores found in file")
        return skill_name, None
    
    results = {"skill_name": skill_name, "task_name": task_name, "metadata": metadata}
    
    # Evaluate VQA metrics
    if mode in ['vqa', 'both']:
        vqa_metrics = evaluate_vqa_metrics(yes_scores, no_scores)
        results["vqa"] = vqa_metrics
        
        print(f"VQA Results:")
        print(f"  Binary Accuracy: {vqa_metrics['binary_acc']:.4f}")
        print(f"  Question Accuracy: {vqa_metrics['question_acc']:.4f}")
    
    # Evaluate Retrieval metrics
    if mode in ['retrieval', 'both']:
        retrieval_scores = compute_retrieval_scores_from_vqa(yes_scores)
        retrieval_metrics = evaluate_retrieval_metrics(retrieval_scores)
        results["retrieval"] = retrieval_metrics
        
        print(f"Retrieval Results:")
        print(f"  Text: {retrieval_metrics['text']:.4f}")
        print(f"  Image: {retrieval_metrics['image']:.4f}")
        print(f"  Group: {retrieval_metrics['group']:.4f}")
    
    return skill_name, results

def format_results(results, mode):
    """Format results for display"""
    output = []
    
    if mode == "vqa":
        output.append("\n" + "="*60)
        output.append("VQA EVALUATION RESULTS")
        output.append("="*60)
        output.append(f"{'Skill':<30} {'Binary Acc':<12} {'Question Acc':<12}")
        output.append("-" * 60)
        
        total_binary = 0.0
        total_question = 0.0
        count = 0
        
        for skill_name, metrics in results.items():
            if metrics and "vqa" in metrics:
                vqa = metrics["vqa"]
                output.append(f"{skill_name:<30} {vqa['binary_acc']:<12.2%} {vqa['question_acc']:<12.2%}")
                total_binary += vqa['binary_acc']
                total_question += vqa['question_acc']
                count += 1
        
        output.append("-" * 60)
        
        if count > 0:
            avg_binary = total_binary / count
            avg_question = total_question / count
            output.append(f"{'Overall Average':<30} {avg_binary:<12.2%} {avg_question:<12.2%}")
    
    elif mode == "retrieval":
        output.append("\n" + "="*70)
        output.append("RETRIEVAL EVALUATION RESULTS")
        output.append("="*70)
        output.append(f"{'Skill':<30} {'Text':<12} {'Image':<12} {'Group':<12}")
        output.append("-" * 70)
        
        # Separate skill tasks vs caption tasks for retrieval only
        skill_tasks_total = {"text": 0.0, "image": 0.0, "group": 0.0, "count": 0}
        caption_tasks_total = {"text": 0.0, "image": 0.0, "group": 0.0, "count": 0}
        
        for skill_name, metrics in results.items():
            if metrics and "retrieval" in metrics:
                retrieval = metrics["retrieval"]
                output.append(f"{skill_name:<30} {retrieval['text']:<12.2%} {retrieval['image']:<12.2%} {retrieval['group']:<12.2%}")
                
                if "complex description" in skill_name.lower() or "caption" in skill_name.lower():
                    caption_tasks_total["text"] += retrieval['text']
                    caption_tasks_total["image"] += retrieval['image']
                    caption_tasks_total["group"] += retrieval['group']
                    caption_tasks_total["count"] += 1
                else:
                    skill_tasks_total["text"] += retrieval['text']
                    skill_tasks_total["image"] += retrieval['image']
                    skill_tasks_total["group"] += retrieval['group']
                    skill_tasks_total["count"] += 1
        
        output.append("-" * 70)
        
        # Show aggregated results for retrieval
        if skill_tasks_total["count"] > 0:
            avg_text = skill_tasks_total["text"] / skill_tasks_total["count"]
            avg_image = skill_tasks_total["image"] / skill_tasks_total["count"]
            avg_group = skill_tasks_total["group"] / skill_tasks_total["count"]
            output.append(f"{'Skill Tasks Average':<30} {avg_text:<12.2%} {avg_image:<12.2%} {avg_group:<12.2%}")
        
        if caption_tasks_total["count"] > 0:
            avg_text = caption_tasks_total["text"] / caption_tasks_total["count"]
            avg_image = caption_tasks_total["image"] / caption_tasks_total["count"]
            avg_group = caption_tasks_total["group"] / caption_tasks_total["count"]
            output.append(f"{'Caption Tasks Average':<30} {avg_text:<12.2%} {avg_image:<12.2%} {avg_group:<12.2%}")
    
    else:  # mode == "both"
        # Format both VQA and retrieval results
        vqa_output = format_results(results, "vqa")
        retrieval_output = format_results(results, "retrieval")
        output.extend(vqa_output)
        output.extend(retrieval_output)
    
    return "\n".join(output)

def save_evaluation_results(results, mode, output_file):
    """Save evaluation results to JSON file"""
    # Prepare summary
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "evaluation_mode": mode,
        "total_skills": len(results),
        "results_by_skill": results
    }
    
    # Compute overall statistics if multiple skills
    if len(results) > 1:
        if mode in ['vqa', 'both']:
            valid_vqa = [r["vqa"] for r in results.values() if r is not None and "vqa" in r]
            if valid_vqa:
                binary_accs = [v["binary_acc"] for v in valid_vqa]
                question_accs = [v["question_acc"] for v in valid_vqa]
                
                summary["overall_vqa_statistics"] = {
                    "mean_binary_acc": float(np.mean(binary_accs)),
                    "std_binary_acc": float(np.std(binary_accs)),
                    "mean_question_acc": float(np.mean(question_accs)),
                    "std_question_acc": float(np.std(question_accs)),
                    "evaluated_skills": len(valid_vqa)
                }
        
        if mode in ['retrieval', 'both']:
            valid_retrieval = [r["retrieval"] for r in results.values() if r is not None and "retrieval" in r]
            if valid_retrieval:
                text_scores = [v["text"] for v in valid_retrieval]
                image_scores = [v["image"] for v in valid_retrieval]
                group_scores = [v["group"] for v in valid_retrieval]
                
                summary["overall_retrieval_statistics"] = {
                    "mean_text": float(np.mean(text_scores)),
                    "std_text": float(np.std(text_scores)),
                    "mean_image": float(np.mean(image_scores)),
                    "std_image": float(np.std(image_scores)),
                    "mean_group": float(np.mean(group_scores)),
                    "std_group": float(np.std(group_scores)),
                    "evaluated_skills": len(valid_retrieval)
                }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Method-agnostic VQA and Retrieval evaluator')
    parser.add_argument('score_files', nargs='+', 
                      help='Score files to evaluate (JSON format from score generation step)')
    parser.add_argument('--mode', type=str, choices=['vqa', 'retrieval', 'both'], default='both',
                      help='Evaluation mode')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file name for results (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"{'='*80}")
    print("METHOD-AGNOSTIC VQA & RETRIEVAL EVALUATION")
    print(f"{'='*80}")
    print(f"Score files to evaluate: {len(args.score_files)}")
    print(f"Evaluation mode: {args.mode}")
    
    # Evaluate each file
    results = {}
    
    for score_file in args.score_files:
        if not Path(score_file).exists():
            print(f"Warning: Score file not found: {score_file}")
            continue
        
        skill_name, metrics = evaluate_single_file(score_file, args.mode)
        
        if metrics:
            results[skill_name] = metrics
    
    # Print summary
    print(format_results(results, args.mode))
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to summarize")
        return
    
    # Print individual results summary
    if args.mode in ['vqa', 'both']:
        print("\nVQA Summary:")
        for skill_name, metrics in valid_results.items():
            if "vqa" in metrics:
                vqa = metrics["vqa"]
                print(f"{skill_name:30s}: Binary={vqa['binary_acc']:.3f}, Question={vqa['question_acc']:.3f}")
        
        if len(valid_results) > 1:
            vqa_metrics = [m["vqa"] for m in valid_results.values() if "vqa" in m]
            if vqa_metrics:
                binary_accs = [v["binary_acc"] for v in vqa_metrics]
                question_accs = [v["question_acc"] for v in vqa_metrics]
                print("-" * 80)
                print(f"{'VQA Overall Average':30s}: Binary={np.mean(binary_accs):.3f} (±{np.std(binary_accs):.3f}), Question={np.mean(question_accs):.3f} (±{np.std(question_accs):.3f})")
    
    if args.mode in ['retrieval', 'both']:
        print("\nRetrieval Summary:")
        for skill_name, metrics in valid_results.items():
            if "retrieval" in metrics:
                ret = metrics["retrieval"]
                print(f"{skill_name:30s}: Text={ret['text']:.3f}, Image={ret['image']:.3f}, Group={ret['group']:.3f}")
        
        if len(valid_results) > 1:
            retrieval_metrics = [m["retrieval"] for m in valid_results.values() if "retrieval" in m]
            if retrieval_metrics:
                text_scores = [v["text"] for v in retrieval_metrics]
                image_scores = [v["image"] for v in retrieval_metrics]
                group_scores = [v["group"] for v in retrieval_metrics]
                print("-" * 80)
                print(f"{'Retrieval Overall Average':30s}: Text={np.mean(text_scores):.3f} (±{np.std(text_scores):.3f}), Image={np.mean(image_scores):.3f} (±{np.std(image_scores):.3f}), Group={np.mean(group_scores):.3f} (±{np.std(group_scores):.3f})")
    
    # Generate output filename if not provided
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"vqa_retrieval_evaluation_{args.mode}_{timestamp}.json"
    
    # Save results
    save_evaluation_results(valid_results, args.mode, output_file)
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()