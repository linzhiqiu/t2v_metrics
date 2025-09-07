#!/usr/bin/env python3
"""
Method-specific script to generate VQA scores using VQAScore models.
This script is specific to VQAScore/LMM evaluation and outputs scores in a standardized format.
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
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

def generate_vqa_scores(data, model_name, video_base_path, checkpoint_name=None, question_template="{} Please only answer Yes or No.", answer_template="Yes"):
    """Generate VQA scores for all samples using the specified model"""
    print(f"Initializing VQAScore model: {model_name}")
    
    if checkpoint_name:
        vqa_scorer = t2v_metrics.VQAScore(model=model_name, checkpoint=checkpoint_name)
    else:
        vqa_scorer = t2v_metrics.VQAScore(model=model_name)
    
    results = []
    
    # Process one sample at a time
    for item in tqdm(data, desc="Computing VQA scores"):
        video_path = item['image']  # Note: using 'image' key for video path
        question = item['question']
        label = item['label']
        
        # Create result entry with metadata
        result_entry = {
            "video_path": video_path,
            "question": question,
            "ground_truth_label": label,
            "method": f"{model_name}" + (f"_{checkpoint_name}" if checkpoint_name else ""),
            "score": None,
            "error": None
        }
        
        # Construct full video path
        full_video_path = os.path.join(video_base_path, video_path)
        
        # Check if video file exists
        if not os.path.exists(full_video_path):
            print(f"Warning: Video not found: {full_video_path}")
            result_entry["error"] = f"Video file not found: {full_video_path}"
            result_entry["score"] = 0.0  # Default score for missing files
            results.append(result_entry)
            continue
        
        try:
            # Compute VQA score for single sample
            score_kwargs = {
                "question_template": question_template,
                "answer_template": answer_template
            }
            
            score = vqa_scorer(images=[full_video_path], texts=[question], **score_kwargs)
            result_entry["score"] = float(score[0].detach().cpu().item())  # Convert tensor to Python float
            
        except Exception as e:
            print(f"Error processing {full_video_path}: {e}")
            result_entry["error"] = str(e)
            result_entry["score"] = 0.0  # Default score for failed samples
        
        results.append(result_entry)
    
    return results

def save_scores(results, output_file, metadata=None):
    """Save scores to JSON file with metadata"""
    output_data = {
        "metadata": metadata or {},
        "scores": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Scores saved to: {output_file}")

def generate_output_filename(model_name, checkpoint_name, split_name):
    """Generate output filename with model, checkpoint, and split names."""
    # Clean model name for filename
    clean_model = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Build filename components
    filename_parts = ["classification_scores", clean_model]
    
    if checkpoint_name:
        clean_checkpoint = checkpoint_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename_parts.append(clean_checkpoint)
    
    filename_parts.append(split_name)
    
    return "_".join(filename_parts) + ".json"

def main():
    parser = argparse.ArgumentParser(description='Generate VQA scores using VQAScore models')
    parser.add_argument('--model', type=str, required=True,
                      help='VQAScore model name (e.g., llava-onevision-qwen2-7b-ov)')
    parser.add_argument('--checkpoint', type=str, required=False,
                      help='Checkpoint name for qwen2.5-vl models (e.g., chancharikm/qwen2.5-vl-7b-cam-motion)')
    parser.add_argument('--data_dir', type=str, default='data/binary_classification',
                      help='Directory containing JSONL files')
    parser.add_argument('--video_dir', type=str, default='data/videos',
                      help='Base directory containing video files')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                      help='Specific split names to evaluate (without .jsonl extension). If not specified, processes all splits.')
    parser.add_argument('--question_template', type=str, default="{} Please only answer Yes or No.",
                      help='Question template for VQA scoring')
    parser.add_argument('--answer_template', type=str, default="Yes",
                      help='Answer template for VQA scoring')
    parser.add_argument('--output_dir', type=str, default='scores',
                      help='Directory to save score files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find JSONL files in data directory
    data_dir = Path(args.data_dir)
    
    if args.splits:
        # Use specified splits
        jsonl_files = []
        for split_name in args.splits:
            jsonl_file = data_dir / f"{split_name}.jsonl"
            if jsonl_file.exists():
                jsonl_files.append((split_name, jsonl_file))
            else:
                print(f"Warning: Split '{split_name}' not found at {jsonl_file}")
    else:
        # Use all JSONL files
        jsonl_files = [(f.stem, f) for f in data_dir.glob('*.jsonl')]
    
    if not jsonl_files:
        if args.splits:
            print(f"No specified splits found in {data_dir}")
        else:
            print(f"No JSONL files found in {data_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Process each split
    for split_name, jsonl_file in jsonl_files:
        print(f"\n{'='*60}")
        print(f"PROCESSING SPLIT: {split_name}")
        print(f"{'='*60}")
        
        # Load data
        data = load_jsonl_data(jsonl_file)
        print(f"Loaded {len(data)} samples from {jsonl_file}")
        
        if len(data) == 0:
            print("Empty dataset, skipping...")
            continue
        
        # Generate scores
        results = generate_vqa_scores(
            data, 
            args.model, 
            args.video_dir,
            args.checkpoint,
            args.question_template,
            args.answer_template
        )
        
        # Create metadata
        metadata = {
            "model_name": args.model,
            "checkpoint": args.checkpoint,
            "split_name": split_name,
            "data_file": str(jsonl_file),
            "video_dir": args.video_dir,
            "question_template": args.question_template,
            "answer_template": args.answer_template,
            "generation_timestamp": datetime.now().isoformat(),
            "method_type": "VQAScore_LMM"
        }
        
        # Generate output filename and save
        output_filename = generate_output_filename(args.model, args.checkpoint, split_name)
        output_path = output_dir / output_filename
        
        save_scores(results, output_path, metadata)

if __name__ == "__main__":
    main()