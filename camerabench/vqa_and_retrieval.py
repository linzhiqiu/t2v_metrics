#!/usr/bin/env python3
"""
Unified script to evaluate VQA and Retrieval tasks on exported data
"""

import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
import t2v_metrics
from tqdm import tqdm
from datetime import datetime

def load_jsonl_data(file_path):
    """Load JSONL data from file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_data_by_skill(data_dir, specific_skill=None):
    """Load data organized by skill and task"""
    data_dir = Path(data_dir)
    skill_data = defaultdict(lambda: defaultdict(list))
    
    if specific_skill:
        # Load specific skill only
        skill_dir = data_dir / specific_skill.replace(" ", "_").replace("/", "_")
        if skill_dir.exists():
            for task_file in skill_dir.glob("*.jsonl"):
                task_name = task_file.stem.replace("_", " ")
                task_data = load_jsonl_data(task_file)
                skill_data[specific_skill][task_name] = task_data
                print(f"Loaded {len(task_data)} samples for {specific_skill} -> {task_name}")
        else:
            print(f"Warning: Skill directory not found: {skill_dir}")
    else:
        # Load all skills
        for skill_dir in data_dir.iterdir():
            if skill_dir.is_dir():
                skill_name = skill_dir.name.replace("_", " ")
                for task_file in skill_dir.glob("*.jsonl"):
                    task_name = task_file.stem.replace("_", " ")
                    task_data = load_jsonl_data(task_file)
                    skill_data[skill_name][task_name] = task_data
                    print(f"Loaded {len(task_data)} samples for {skill_name} -> {task_name}")
    
    return skill_data

def compute_scores(samples, model, question_template="{}"):
    """Compute scores for samples (used for both VQA and retrieval evaluation)"""
    yes_scores = []
    no_scores = []
    
    for sample in tqdm(samples, desc="Computing scores"):
        pos_video = sample["pos_video"]
        neg_video = sample["neg_video"]
        
        if "pos_question" in sample:
            pos_question = sample["pos_question"]
            neg_question = sample["neg_question"]
        else:
            # Fallback for retrieval-style data
            pos_question = sample["pos_text"]
            neg_question = sample["neg_text"]
        
        try:
            # Use question_template and answer_template like original scripts
            yes_kwargs = {"question_template": question_template, "answer_template": "Yes"}
            no_kwargs = {"question_template": question_template, "answer_template": "No"}
            
            # Compute scores for all 4 combinations with "Yes" answer
            yes_pos_text_pos_image = model(images=[pos_video], texts=[pos_question], **yes_kwargs)[0].detach().cpu().item()
            yes_pos_text_neg_image = model(images=[neg_video], texts=[pos_question], **yes_kwargs)[0].detach().cpu().item()
            yes_neg_text_pos_image = model(images=[pos_video], texts=[neg_question], **yes_kwargs)[0].detach().cpu().item()
            yes_neg_text_neg_image = model(images=[neg_video], texts=[neg_question], **yes_kwargs)[0].detach().cpu().item()
            
            # Compute scores for all 4 combinations with "No" answer
            no_pos_text_pos_image = model(images=[pos_video], texts=[pos_question], **no_kwargs)[0].detach().cpu().item()
            no_pos_text_neg_image = model(images=[neg_video], texts=[pos_question], **no_kwargs)[0].detach().cpu().item()
            no_neg_text_pos_image = model(images=[pos_video], texts=[neg_question], **no_kwargs)[0].detach().cpu().item()
            no_neg_text_neg_image = model(images=[neg_video], texts=[neg_question], **no_kwargs)[0].detach().cpu().item()
            
            # Store in matrix format [pos_text_pos_image, pos_text_neg_image, neg_text_pos_image, neg_text_neg_image]
            yes_scores.append([yes_pos_text_pos_image, yes_pos_text_neg_image, yes_neg_text_pos_image, yes_neg_text_neg_image])
            no_scores.append([no_pos_text_pos_image, no_pos_text_neg_image, no_neg_text_pos_image, no_neg_text_neg_image])
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            yes_scores.append([0.0, 0.0, 0.0, 0.0])
            no_scores.append([0.0, 0.0, 0.0, 0.0])
    
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

def evaluate_skill_data_both_modes(skill_data, model, question_template="{}"):
    """Evaluate skill data for both VQA and retrieval modes (compute scores once)"""
    all_samples = []
    for task_name, task_samples in skill_data.items():
        all_samples.extend(task_samples)
    
    if len(all_samples) == 0:
        return {
            "vqa": {"binary_acc": 0.0, "question_acc": 0.0},
            "retrieval": {"text": 0.0, "image": 0.0, "group": 0.0}
        }
    
    # Compute scores once
    yes_scores, no_scores = compute_scores(all_samples, model, question_template)
    
    # Evaluate both modes using the same scores
    vqa_metrics = evaluate_vqa_metrics(yes_scores, no_scores)
    retrieval_scores = compute_retrieval_scores_from_vqa(yes_scores)
    retrieval_metrics = evaluate_retrieval_metrics(retrieval_scores)
    
    return {
        "vqa": vqa_metrics,
        "retrieval": retrieval_metrics
    }

def format_results(results, mode, specific_skill=None):
    """Format results for display"""
    output = []
    
    if mode == "vqa":
        output.append("\n" + "="*60)
        output.append("VQA EVALUATION RESULTS")
        output.append("="*60)
        output.append(f"{'Skill':<30} {'Binary Acc':<12} {'Question Acc':<12}")
        output.append("-" * 60)
        
        # For VQA, treat all skills equally - no separation
        total_binary = 0.0
        total_question = 0.0
        count = 0
        
        for skill_name, metrics in results.items():
            output.append(f"{skill_name:<30} {metrics['binary_acc']:<12.2%} {metrics['question_acc']:<12.2%}")
            total_binary += metrics['binary_acc']
            total_question += metrics['question_acc']
            count += 1
        
        output.append("-" * 60)
        
        if not specific_skill and count > 0:
            # Show overall average for all skills
            avg_binary = total_binary / count
            avg_question = total_question / count
            output.append(f"{'Overall Average':<30} {avg_binary:<12.2%} {avg_question:<12.2%}")
    
    else:  # retrieval
        output.append("\n" + "="*70)
        output.append("RETRIEVAL EVALUATION RESULTS")
        output.append("="*70)
        output.append(f"{'Skill':<30} {'Text':<12} {'Image':<12} {'Group':<12}")
        output.append("-" * 70)
        
        # Separate skill tasks vs caption tasks for retrieval only
        skill_tasks_total = {"text": 0.0, "image": 0.0, "group": 0.0, "count": 0}
        caption_tasks_total = {"text": 0.0, "image": 0.0, "group": 0.0, "count": 0}
        
        for skill_name, metrics in results.items():
            output.append(f"{skill_name:<30} {metrics['text']:<12.2%} {metrics['image']:<12.2%} {metrics['group']:<12.2%}")
            
            if "complex description" in skill_name.lower() or "caption" in skill_name.lower():
                caption_tasks_total["text"] += metrics['text']
                caption_tasks_total["image"] += metrics['image']
                caption_tasks_total["group"] += metrics['group']
                caption_tasks_total["count"] += 1
            else:
                skill_tasks_total["text"] += metrics['text']
                skill_tasks_total["image"] += metrics['image']
                skill_tasks_total["group"] += metrics['group']
                skill_tasks_total["count"] += 1
        
        output.append("-" * 70)
        
        if not specific_skill:
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
    
    return "\n".join(output)

def generate_output_filename(model_name, checkpoint_name, mode, skill=None):
    """Generate output filename with model, checkpoint, mode, skill, and timestamp"""
    # Clean model name for filename (replace problematic characters)
    clean_model = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename components
    filename_parts = ["unified_eval_results", clean_model]
    
    if checkpoint_name:
        clean_checkpoint = checkpoint_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename_parts.append(clean_checkpoint)
    
    filename_parts.append(mode)
    
    if skill:
        clean_skill = skill.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_')
        filename_parts.append(clean_skill)
    
    filename_parts.append(timestamp)
    
    return "_".join(filename_parts) + ".json"

def main():
    parser = argparse.ArgumentParser(description='Unified VQA and Retrieval Evaluation')
    parser.add_argument('--model', type=str, required=True,
                      help='Model name (e.g., llava-onevision-qwen2-7b-ov)')
    parser.add_argument('--checkpoint', type=str, required=False,
                      help='Checkpoint name for qwen2.5-vl models (e.g., chancharikm/qwen2.5-vl-7b-cam-motion)')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing exported data')
    parser.add_argument('--mode', type=str, choices=['vqa', 'retrieval', 'both'], default='both',
                      help='Evaluation mode')
    parser.add_argument('--skill', type=str, default=None,
                      help='Specific skill to evaluate (e.g., "Motion & Steadiness")')
    parser.add_argument('--question_template', type=str, 
                      default="{} Please only answer Yes or No.",
                      help='Question template for VQA evaluation')
    parser.add_argument('--output_file', type=str, default=None,
                      help='File to save results (optional, auto-generated if not provided)')
    
    args = parser.parse_args()
    
    print(f"Initializing model: {args.model}")
    if args.checkpoint:
        model = t2v_metrics.VQAScore(model=args.model, checkpoint=args.checkpoint)
    else:
        model = t2v_metrics.VQAScore(model=args.model)
    
    data_dir = Path(args.data_dir)
    results = {}
    
    # Load data once and use for both modes if needed
    vqa_dir = data_dir / "vqa_and_retrieval"
    if not vqa_dir.exists():
        print(f"VQA data directory not found: {vqa_dir}")
        return
    
    skill_data = load_data_by_skill(vqa_dir, args.skill)
    
    if args.mode in ['vqa', 'both']:
        print("\n" + "="*50)
        print("EVALUATING VQA")
        print("="*50)
        
        vqa_results = {}
        for skill_name, skill_data_dict in skill_data.items():
            print(f"\nEvaluating VQA for skill: {skill_name}")
            metrics = evaluate_skill_data_both_modes(skill_data_dict, model, args.question_template)
            vqa_results[skill_name] = metrics["vqa"]
        
        results['vqa'] = vqa_results
        print(format_results(vqa_results, "vqa", args.skill))
    
    if args.mode in ['retrieval', 'both']:
        print("\n" + "="*50)
        print("EVALUATING RETRIEVAL")
        print("="*50)
        
        retrieval_results = {}
        for skill_name, skill_data_dict in skill_data.items():
            print(f"\nEvaluating Retrieval for skill: {skill_name}")
            if args.mode == 'both':
                # Use existing metrics if we already computed them
                metrics = evaluate_skill_data_both_modes(skill_data_dict, model, args.question_template)
                retrieval_results[skill_name] = metrics["retrieval"]
            else:
                # Compute fresh if only doing retrieval
                metrics = evaluate_skill_data_both_modes(skill_data_dict, model, args.question_template)
                retrieval_results[skill_name] = metrics["retrieval"]
        
        results['retrieval'] = retrieval_results
        print(format_results(retrieval_results, "retrieval", args.skill))
    
    # Generate output filename if not provided
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_filename = generate_output_filename(args.model, args.checkpoint, args.mode, args.skill)
        output_path = Path(output_filename)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()