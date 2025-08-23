#!/usr/bin/env python3
"""
Fixed parallel evaluation - explicitly assign one GPU per worker
"""

import json
import os
import subprocess
import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
import torch
from datetime import datetime

def create_single_gpu_worker():
    """Create a worker script that uses ONLY the assigned GPU"""
    worker_code = '''#!/usr/bin/env python3
import json
import os
import sys

def main():
    gpu_id = int(sys.argv[1])
    chunk_file = sys.argv[2] 
    model_name = sys.argv[3]
    checkpoint_name = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "None" else None
    result_file = sys.argv[5]
    
    print(f"Worker starting: GPU {gpu_id}, chunk {chunk_file}")
    
    # FORCE this process to see ONLY the assigned GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import after setting CUDA_VISIBLE_DEVICES
    import t2v_metrics
    import torch
    from tqdm import tqdm
    
    print(f"Worker GPU {gpu_id}: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Worker GPU {gpu_id}: Available CUDA devices = {torch.cuda.device_count()}")
    
    # Load data
    print(f"Worker GPU {gpu_id}: Loading data...")
    data = []
    try:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Worker GPU {gpu_id}: JSON error on line {line_num}: {e}")
                    print(f"Worker GPU {gpu_id}: Problematic line: {repr(line[:100])}")
                    raise
    except Exception as e:
        print(f"Worker GPU {gpu_id}: Error loading data: {e}")
        with open(result_file, 'w') as f:
            json.dump({'gpu_id': gpu_id, 'scores': [], 'labels': [], 'error': f"Data loading error: {str(e)}"}, f)
        return
    
    print(f"Worker GPU {gpu_id}: Loaded {len(data)} samples")
    
    # Initialize model
    print(f"Worker GPU {gpu_id}: Initializing model {model_name}...")
    try:
        if checkpoint_name:
            scorer = t2v_metrics.VQAScore(model=model_name, checkpoint=checkpoint_name)
        else:
            scorer = t2v_metrics.VQAScore(model=model_name)
        print(f"Worker GPU {gpu_id}: Model loaded successfully")
    except Exception as e:
        print(f"Worker GPU {gpu_id}: Error loading model: {e}")
        with open(result_file, 'w') as f:
            json.dump({'gpu_id': gpu_id, 'scores': [], 'labels': [], 'error': str(e)}, f)
        return
    
    # Process data
    print(f"Worker GPU {gpu_id}: Starting evaluation...")
    scores = []
    labels = []
    
    pbar = tqdm(data, desc=f"GPU {gpu_id}")
    for item in pbar:
        try:
            score = scorer(
                images=[item['image']], 
                texts=[item['question']],
                question_template="{} Please only answer Yes or No.",
                answer_template="Yes"
            )
            scores.append(score[0].detach().cpu().item())
            labels.append(1 if item['label'].lower() == 'yes' else 0)
        except Exception as e:
            print(f"Worker GPU {gpu_id}: Error processing {item['image']}: {e}")
            scores.append(0.0)
            labels.append(1 if item['label'].lower() == 'yes' else 0)
    
    pbar.close()
    
    # Save results
    results = {
        'gpu_id': gpu_id,
        'scores': scores,
        'labels': labels,
        'num_processed': len(scores)
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f)
    
    print(f"Worker GPU {gpu_id}: COMPLETED! Processed {len(scores)} samples, saved to {result_file}")

if __name__ == "__main__":
    main()
'''
    
    with open('single_gpu_worker.py', 'w') as f:
        f.write(worker_code)
    print("Created single_gpu_worker.py")

def split_data(data, num_chunks):
    """Split data into roughly equal chunks"""
    if num_chunks <= 0:
        return [data]
        
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    
    chunks = []
    start = 0
    for i in range(num_chunks):
        size = chunk_size + (1 if i < remainder else 0)
        if size > 0:
            chunks.append(data[start:start+size])
        start += size
    
    return chunks

def run_parallel_evaluation(data, model_name, checkpoint_name):
    """Run evaluation using separate processes, each locked to one GPU"""
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")
    
    print(f"\n{'='*60}")
    print(f"PARALLEL EVALUATION SETUP")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")
    print(f"Available GPUs: {num_gpus}")
    
    # Create worker script
    create_single_gpu_worker()
    
    # Split data into chunks
    chunks = split_data(data, num_gpus)
    print(f"Data chunks: {[len(c) for c in chunks]}")
    
    # Create chunk files and start processes
    processes = []
    temp_files = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            print(f"Skipping GPU {i} (empty chunk)")
            continue
            
        # Save chunk to file
        chunk_file = f'chunk_gpu_{i}.jsonl'
        result_file = f'result_gpu_{i}.json'
        
        print(f"Writing chunk file: {chunk_file}")
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for item in chunk:
                # Ensure proper JSONL format - each line is a complete JSON object
                json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
                f.write(json_line + '\n')
        
        print(f"Chunk file {chunk_file} written with {len(chunk)} items")
        
        temp_files.append((chunk_file, result_file))
        
        # Start worker process with explicit GPU assignment
        cmd = [
            sys.executable, 'single_gpu_worker.py',
            str(i),  # GPU ID
            chunk_file,  # Input chunk
            model_name,  # Model name
            checkpoint_name if checkpoint_name else "None",  # Checkpoint
            result_file  # Output file
        ]
        
        print(f"Starting worker for GPU {i}: {len(chunk)} samples")
        print(f"Command: {' '.join(cmd)}")
        
        # Start the process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append((proc, i))
    
    print(f"\\nStarted {len(processes)} worker processes")
    print(f"{'='*60}")
    print("WORKERS RUNNING - Monitor progress below:")
    print(f"{'='*60}")
    
    # Wait for all processes and collect output
    for proc, gpu_id in processes:
        print(f"\\n--- GPU {gpu_id} Output ---")
        for line in proc.stdout:
            print(f"[GPU {gpu_id}] {line.rstrip()}")
        
        proc.wait()
        print(f"--- GPU {gpu_id} Finished (exit code: {proc.returncode}) ---")
    
    print(f"\\n{'='*60}")
    print("ALL WORKERS COMPLETED - COLLECTING RESULTS")
    print(f"{'='*60}")
    
    # Collect results
    all_scores = []
    all_labels = []
    
    for chunk_file, result_file in temp_files:
        try:
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                all_scores.extend(data['scores'])
                all_labels.extend(data['labels'])
                
                print(f"✓ GPU {data['gpu_id']}: {data['num_processed']} samples processed")
                
                if 'error' in data:
                    print(f"  ⚠ GPU {data['gpu_id']} had error: {data['error']}")
            else:
                print(f"✗ Missing result file: {result_file}")
                
        except Exception as e:
            print(f"✗ Error reading {result_file}: {e}")
        
        # Clean up temp files
        for temp_file in [chunk_file, result_file]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
    
    # Clean up worker script
    try:
        os.remove('single_gpu_worker.py')
    except:
        pass
    
    return np.array(all_scores), np.array(all_labels)

def compute_map(scores, labels):
    """Compute Average Precision for binary classification"""
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels")
        return 0.0
    
    scores = np.where(np.isfinite(scores), scores, -1e10)
    ap = average_precision_score(labels, scores)
    return ap

def main():
    parser = argparse.ArgumentParser(description='Fixed parallel VQAScore evaluation')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--checkpoint', help='Checkpoint name (optional)')
    parser.add_argument('--data_dir', default='data/binary_classification', help='Data directory')
    parser.add_argument('--splits', nargs='+', help='Specific splits to evaluate')
    
    args = parser.parse_args()
    
    print(f"GPU check: {torch.cuda.device_count()} GPUs available")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Find JSONL files
    data_dir = Path(args.data_dir)
    if args.splits:
        files = [data_dir / f"{s}.jsonl" for s in args.splits if (data_dir / f"{s}.jsonl").exists()]
    else:
        files = list(data_dir.glob('*.jsonl'))
    
    if not files:
        print("No JSONL files found!")
        return
    
    print(f"Found {len(files)} files to process: {[f.name for f in files]}")
    
    # Process each split
    results = {}
    
    for file_path in files:
        split_name = file_path.stem
        print(f"\\n{'='*80}")
        print(f"PROCESSING SPLIT: {split_name}")
        print(f"{'='*80}")
        
        # Load data
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(data)} samples from {file_path}")
        
        if len(data) == 0:
            print("Empty dataset, skipping...")
            results[split_name] = 0.0
            continue
        
        # Run parallel evaluation
        scores, labels = run_parallel_evaluation(data, args.model, args.checkpoint)
        
        if len(scores) == 0:
            print("No scores computed!")
            results[split_name] = 0.0
            continue
        
        # Compute mAP
        map_score = compute_map(scores, labels)
        results[split_name] = map_score
        
        print(f"\\n{'='*60}")
        print(f"SPLIT {split_name} RESULTS:")
        print(f"  Samples processed: {len(scores)}")
        print(f"  Positive samples: {np.sum(labels == 1)}")
        print(f"  Negative samples: {np.sum(labels == 0)}")
        print(f"  Average Precision (mAP): {map_score:.4f}")
        print(f"{'='*60}")
    
    # Final summary
    print(f"\\n{'='*80}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    for split_name, map_score in results.items():
        print(f"{split_name:30s}: mAP = {map_score:.4f}")
    
    if results:
        overall_map = np.mean(list(results.values()))
        print("-" * 80)
        print(f"{'Overall Average':30s}: mAP = {overall_map:.4f}")
        
        # Save results
        clean_model = args.model.replace('/', '_').replace('\\', '_').replace(':', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = ["vqascore_results", clean_model]
        
        if args.checkpoint:
            clean_checkpoint = args.checkpoint.replace('/', '_').replace('\\', '_').replace(':', '_')
            filename_parts.append(clean_checkpoint)
        
        filename_parts.append(timestamp)
        output_file = "_".join(filename_parts) + ".json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\nResults saved to: {output_file}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()