#!/usr/bin/env python3
"""
Script to evaluate VQAScore on binary classification tasks and compute mAP
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import t2v_metrics
from tqdm import tqdm
from datetime import datetime

def load_json_data(file_path):
    """Load JSONL data from file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def compute_vqa_scores(data, model_name, checkpoint_name):
    """Compute VQA scores for all samples"""
    print(f"Initializing VQAScore model: {model_name}")

    if checkpoint_name:
        vqa_scorer = t2v_metrics.VQAScore(model=model_name, checkpoint=checkpoint_name)
    else:
        vqa_scorer = t2v_metrics.VQAScore(model=model_name)
    
    scores = []
    labels = []
    
    # Process one sample at a time
    for item in tqdm(data, desc="Computing VQA scores"):
        video_path = item['image']  # Note: using 'image' key for video path
        question = item['question']
        label = item['label']
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        try:
            # Use question_template and answer_template like original scripts
            score_kwargs = {
                "question_template": "{} Please only answer Yes or No.",
                "answer_template": "Yes"
            }
            
            # Compute VQA score for single sample
            score = vqa_scorer(images=[video_path], texts=[question], **score_kwargs)
            scores.append(score[0].detach().cpu().item())  # Convert tensor to Python float
            labels.append(1 if label.lower() == 'yes' else 0)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Add placeholder score for failed sample
            scores.append(0.0)
            labels.append(1 if label.lower() == 'yes' else 0)
    
    return np.array(scores), np.array(labels)

def compute_map(scores, labels):
    """Compute Average Precision for binary classification"""
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels")
        return 0.0
    
    # Ensure scores are finite (replace NaN/inf with very low values like original)
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Compute Average Precision
    ap = average_precision_score(labels, scores)
    return ap

def evaluate_split(json_file, model_name, checkpoint_name):
    """Evaluate a single split and return mAP"""
    print(f"\nEvaluating {json_file}")
    
    # Load data
    data = load_json_data(json_file)
    print(f"Loaded {len(data)} samples")
    
    if len(data) == 0:
        return 0.0
    
    # Compute VQA scores
    scores, labels = compute_vqa_scores(data, model_name, checkpoint_name)
    
    if len(scores) == 0:
        print("No valid scores computed")
        return 0.0
    
    # Compute mAP
    map_score = compute_map(scores, labels)
    
    # Print some statistics
    print(f"Samples processed: {len(scores)}")
    print(f"Positive samples: {np.sum(labels == 1)}")
    print(f"Negative samples: {np.sum(labels == 0)}")
    print(f"Average Precision (mAP): {map_score:.4f}")
    
    return map_score

def generate_output_filename(model_name, checkpoint_name):
    """Generate output filename with model, checkpoint, and timestamp"""
    # Clean model name for filename (replace problematic characters)
    clean_model = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename components
    filename_parts = ["vqascore_results", clean_model]
    
    if checkpoint_name:
        clean_checkpoint = checkpoint_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename_parts.append(clean_checkpoint)
    
    filename_parts.append(timestamp)
    
    return "_".join(filename_parts) + ".json"

def main():
    parser = argparse.ArgumentParser(description='Evaluate VQAScore on binary classification tasks using JSONL files')
    parser.add_argument('--model', type=str, required=True,
                      help='VQAScore model name (e.g., llava-onevision-qwen2-7b-ov)')
    parser.add_argument('--checkpoint', type=str, required=False,
                      help='Checkpoint name for qwen2.5-vl models (e.g., chancharikm/qwen2.5-vl-7b-cam-motion)')
    parser.add_argument('--data_dir', type=str, default='data/binary_classification',
                      help='Directory containing JSONL files')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                      help='Specific split names to evaluate (without .jsonl extension). If not specified, evaluates all splits.')
    
    args = parser.parse_args()
    
    # Find JSONL files in data directory
    data_dir = Path(args.data_dir)
    
    if args.splits:
        # Use specified splits
        json_files = []
        for split_name in args.splits:
            jsonl_file = data_dir / f"{split_name}.jsonl"
            if jsonl_file.exists():
                json_files.append(jsonl_file)
            else:
                print(f"Warning: Split '{split_name}' not found at {jsonl_file}")
    else:
        # Use all JSONL files
        json_files = list(data_dir.glob('*.jsonl'))
    
    if not json_files:
        if args.splits:
            print(f"No specified splits found in {data_dir}")
        else:
            print(f"No JSONL files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} JSONL files to evaluate")
    
    # Evaluate each split
    results = {}
    for json_file in json_files:
        split_name = json_file.stem  # filename without extension
        map_score = evaluate_split(json_file, args.model, args.checkpoint)
        results[split_name] = map_score
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for split_name, map_score in results.items():
        print(f"{split_name:30s}: mAP = {map_score:.4f}")
    
    # Overall average
    if results:
        overall_map = np.mean(list(results.values()))
        print("-"*50)
        print(f"{'Overall Average':30s}: mAP = {overall_map:.4f}")
    
    # Generate unique output filename
    output_file = generate_output_filename(args.model, args.checkpoint)
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()