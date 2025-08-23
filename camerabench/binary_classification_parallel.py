#!/usr/bin/env python3
"""
Parallelized script to evaluate VQAScore on binary classification tasks and compute mAP
Distributes work across multiple GPUs for faster evaluation
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
import torch
import multiprocessing as mp
from functools import partial
import time

def load_json_data(file_path):
    """Load JSONL data from file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def chunk_data(data, num_chunks):
    """Split data into roughly equal chunks"""
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        # Add one extra item to the first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(data[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks

def compute_vqa_scores_gpu(data_chunk, model_name, checkpoint_name, gpu_id, results_file):
    """Compute VQA scores for a chunk of data on a specific GPU"""
    try:
        # Set CUDA device for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        print(f"GPU {gpu_id}: Initializing VQAScore model: {model_name}")
        
        # Initialize model
        if checkpoint_name:
            vqa_scorer = t2v_metrics.VQAScore(model=model_name, checkpoint=checkpoint_name)
        else:
            vqa_scorer = t2v_metrics.VQAScore(model=model_name)
        
        scores = []
        labels = []
        
        # Process chunk
        print(f"GPU {gpu_id}: Processing {len(data_chunk)} samples")
        for i, item in enumerate(data_chunk):
            video_path = item['image']  # Note: using 'image' key for video path
            question = item['question']
            label = item['label']
            
            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"GPU {gpu_id}: Warning: Video not found: {video_path}")
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
                print(f"GPU {gpu_id}: Error processing {video_path}: {e}")
                # Add placeholder score for failed sample
                scores.append(0.0)
                labels.append(1 if label.lower() == 'yes' else 0)
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"GPU {gpu_id}: Processed {i+1}/{len(data_chunk)} samples")
        
        # Save results to temporary file
        result_data = {
            'scores': scores,
            'labels': labels,
            'gpu_id': gpu_id,
            'num_processed': len(scores)
        }
        
        with open(results_file, 'w') as f:
            json.dump(result_data, f)
        
        print(f"GPU {gpu_id}: Completed processing {len(data_chunk)} samples, saved to {results_file}")
        return True
        
    except Exception as e:
        print(f"GPU {gpu_id}: Fatal error: {e}")
        # Save empty results
        result_data = {
            'scores': [],
            'labels': [],
            'gpu_id': gpu_id,
            'num_processed': 0,
            'error': str(e)
        }
        
        with open(results_file, 'w') as f:
            json.dump(result_data, f)
        
        return False

def compute_vqa_scores_parallel(data, model_name, checkpoint_name):
    """Compute VQA scores in parallel across available GPUs"""
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available for parallel processing")
    
    print(f"Using {num_gpus} GPUs for parallel evaluation")
    
    # Split data into chunks
    data_chunks = chunk_data(data, num_gpus)
    print(f"Split {len(data)} samples into {len(data_chunks)} chunks: {[len(chunk) for chunk in data_chunks]}")
    
    # Save data chunks to temporary files to avoid pickling issues
    chunk_files = []
    result_files = []
    
    for i, chunk in enumerate(data_chunks):
        if len(chunk) > 0:  # Only create files for non-empty chunks
            # Save chunk data
            chunk_file = f"temp_chunk_{i}_{os.getpid()}.jsonl"
            with open(chunk_file, 'w') as f:
                for item in chunk:
                    f.write(json.dumps(item) + '\n')
            chunk_files.append((chunk_file, i))
            
            # Prepare result file name
            result_file = f"temp_results_{i}_{os.getpid()}.json"
            result_files.append(result_file)
    
    # Create and start worker processes
    processes = []
    
    for (chunk_file, gpu_id), result_file in zip(chunk_files, result_files):
        p = mp.Process(
            target=worker_process,
            args=(chunk_file, model_name, checkpoint_name, gpu_id, result_file)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    print("Waiting for all GPU processes to complete...")
    for p in processes:
        p.join()
    
    # Collect results from temporary files
    all_scores = []
    all_labels = []
    
    for result_file in result_files:
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                all_scores.extend(result_data['scores'])
                all_labels.extend(result_data['labels'])
                
                print(f"Collected {result_data['num_processed']} results from GPU {result_data['gpu_id']}")
                
                if 'error' in result_data:
                    print(f"GPU {result_data['gpu_id']} reported error: {result_data['error']}")
                
                # Clean up result file
                os.remove(result_file)
            except Exception as e:
                print(f"Error reading result file {result_file}: {e}")
    
    # Clean up chunk files
    for chunk_file, _ in chunk_files:
        try:
            os.remove(chunk_file)
        except:
            pass
    
    return np.array(all_scores), np.array(all_labels)

def worker_process(chunk_file, model_name, checkpoint_name, gpu_id, result_file):
    """Worker process that loads data from file and processes it"""
    try:
        # Load data chunk from file
        data_chunk = load_json_data(chunk_file)
        
        # Process the chunk
        compute_vqa_scores_gpu(data_chunk, model_name, checkpoint_name, gpu_id, result_file)
        
    except Exception as e:
        print(f"Worker process for GPU {gpu_id} failed: {e}")
        # Save empty results
        result_data = {
            'scores': [],
            'labels': [],
            'gpu_id': gpu_id,
            'num_processed': 0,
            'error': str(e)
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f)

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

def evaluate_split(json_file, model_name, checkpoint_name, use_parallel=True):
    """Evaluate a single split and return mAP"""
    print(f"\nEvaluating {json_file}")
    
    # Load data
    data = load_json_data(json_file)
    print(f"Loaded {len(data)} samples")
    
    if len(data) == 0:
        return 0.0
    
    # Compute VQA scores
    if use_parallel and torch.cuda.device_count() > 1:
        scores, labels = compute_vqa_scores_parallel(data, model_name, checkpoint_name)
    else:
        print("Using single GPU evaluation (parallel disabled or only 1 GPU available)")
        # Fall back to original single-GPU implementation
        if torch.cuda.is_available():
            if checkpoint_name:
                vqa_scorer = t2v_metrics.VQAScore(model=model_name, checkpoint=checkpoint_name)
            else:
                vqa_scorer = t2v_metrics.VQAScore(model=model_name)
        
            scores = []
            labels = []
            
            for item in tqdm(data, desc="Computing VQA scores"):
                video_path = item['image']
                question = item['question']
                label = item['label']
                
                if not os.path.exists(video_path):
                    print(f"Warning: Video not found: {video_path}")
                    continue
                
                try:
                    score_kwargs = {
                        "question_template": "{} Please only answer Yes or No.",
                        "answer_template": "Yes"
                    }
                    
                    score = vqa_scorer(images=[video_path], texts=[question], **score_kwargs)
                    scores.append(score[0].detach().cpu().item())
                    labels.append(1 if label.lower() == 'yes' else 0)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    scores.append(0.0)
                    labels.append(1 if label.lower() == 'yes' else 0)
            
            scores, labels = np.array(scores), np.array(labels)
    
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
    parser = argparse.ArgumentParser(description='Evaluate VQAScore on binary classification tasks using JSONL files (parallelized)')
    parser.add_argument('--model', type=str, required=True,
                      help='VQAScore model name (e.g., llava-onevision-qwen2-7b-ov)')
    parser.add_argument('--checkpoint', type=str, required=False,
                      help='Checkpoint name for qwen2.5-vl models (e.g., chancharikm/qwen2.5-vl-7b-cam-motion)')
    parser.add_argument('--data_dir', type=str, default='data/binary_classification',
                      help='Directory containing JSONL files')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                      help='Specific split names to evaluate (without .jsonl extension). If not specified, evaluates all splits.')
    parser.add_argument('--no_parallel', action='store_true',
                      help='Disable parallel processing (use single GPU)')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA device(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available")
        return
    
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
    use_parallel = not args.no_parallel
    
    for json_file in json_files:
        split_name = json_file.stem  # filename without extension
        map_score = evaluate_split(json_file, args.model, args.checkpoint, use_parallel)
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
    # Set multiprocessing start method to spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()