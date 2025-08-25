#!/usr/bin/env python3
"""
Method-specific script to generate captions using various models.
This script is specific to model implementations and outputs captions in a standardized format.
"""

import json
import os
import argparse
import t2v_metrics
from tqdm import tqdm
from datetime import datetime
import time

def parse_model_spec(model_spec):
    """
    Parse model specification string.
    
    Args:
        model_spec: String in format "model_name" or "model_name:checkpoint"
        
    Returns:
        Tuple of (model_name, checkpoint_name)
    """
    if ':' in model_spec:
        model_name, checkpoint_name = model_spec.split(':', 1)
        return model_name.strip(), checkpoint_name.strip()
    else:
        return model_spec.strip(), None

def process_videos_with_model(model_spec, videos, questions):
    """
    Process all videos with a single model instance.
    
    Args:
        model_spec: Model specification string (model_name or model_name:checkpoint)
        videos: List of video paths
        questions: List of questions corresponding to videos
        
    Returns:
        List of result entries with metadata
    """
    model_name, checkpoint_name = parse_model_spec(model_spec)
    
    print(f"\nLoading model: {model_name}")
    if checkpoint_name:
        print(f"Using checkpoint: {checkpoint_name}")
    
    results = []
    method_name = f"{model_name}" + (f"_{checkpoint_name}" if checkpoint_name else "")
    
    try:
        # Initialize the model once
        if 'gemini' in model_name:
            score_model = t2v_metrics.get_score_model(model=model_name, api_key='api_key')
        elif 'gpt' in model_name:
            score_model = t2v_metrics.get_score_model(model=model_name, api_key='api_key')
        else:
            # For other models, pass checkpoint parameter if provided
            if checkpoint_name:
                score_model = t2v_metrics.VQAScore(model=model_name, checkpoint=checkpoint_name)
            else:
                score_model = t2v_metrics.VQAScore(model=model_name)
        
        # Process all videos with this model
        for i, (video, question) in enumerate(tqdm(zip(videos, questions), total=len(videos), desc=f"Processing with {model_name}")):
            result_entry = {
                "sample_id": str(i),
                "video": video,
                "question": question,
                "method": method_name,
                "caption": None,
                "error": None
            }
            
            try:
                # Generate the caption
                response = score_model.model.generate(images=[video], texts=[question])
                caption = response[0] if isinstance(response, list) else response
                result_entry["caption"] = caption
            except Exception as e:
                error_msg = f"Error processing video {i}: {str(e)}"
                print(error_msg)
                result_entry["error"] = str(e)
                result_entry["caption"] = ""  # Empty caption for failed samples
                
            results.append(result_entry)
                
        return results
        
    except Exception as e:
        print(f"Fatal error with model {model_name}: {str(e)}")
        # Return error entries for all videos if model initialization fails
        for i, (video, question) in enumerate(zip(videos, questions)):
            results.append({
                "sample_id": str(i),
                "video": video,
                "question": question,
                "method": method_name,
                "caption": "",
                "error": f"Model initialization failed: {str(e)}"
            })
        return results

def save_captions(results, output_file, metadata=None):
    """Save captions to JSON file with metadata"""
    output_data = {
        "metadata": metadata or {},
        "captions": results,
        "total_samples": len(results),
        "successful_samples": len([r for r in results if r["error"] is None]),
        "failed_samples": len([r for r in results if r["error"] is not None])
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Captions saved to: {output_file}")
    print(f"Total samples: {output_data['total_samples']}")
    print(f"Successful: {output_data['successful_samples']}")
    print(f"Failed: {output_data['failed_samples']}")

def generate_output_filename(model_spec):
    """Generate output filename with model, checkpoint, and timestamp"""
    model_name, checkpoint_name = parse_model_spec(model_spec)
    
    # Clean model name for filename
    clean_model = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename components
    filename_parts = ["captions", clean_model]
    
    if checkpoint_name:
        clean_checkpoint = checkpoint_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename_parts.append(clean_checkpoint)
    
    filename_parts.append(timestamp)
    
    return "_".join(filename_parts) + ".json"

def main():
    parser = argparse.ArgumentParser(description="Generate captions using different models")
    parser.add_argument("--input", type=str, default="~/test_caption.json", 
                        help="Path to input JSON file")
    parser.add_argument("--output_dir", type=str, default="captions", 
                        help="Directory to save caption files")
    parser.add_argument("--sample_size", type=int, default=100, 
                        help="Number of items to sample")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=["qwen2.5-vl-7b"],
                        help="List of model specifications. Format: model_name or model_name:checkpoint")
    
    # Backwards compatibility for single model
    parser.add_argument("--model", type=str, 
                        help="Single model name (for backwards compatibility)")
    parser.add_argument("--checkpoint", type=str, 
                        help="Single model checkpoint (for backwards compatibility)")
    
    args = parser.parse_args()
    
    # Handle backwards compatibility for single model arguments
    if args.model:
        if args.checkpoint:
            single_model_spec = f"{args.model}:{args.checkpoint}"
        else:
            single_model_spec = args.model
        args.models = [single_model_spec]
        print(f"Using single model specification: {single_model_spec}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Expand the tilde in the path
    input_path = os.path.expanduser(args.input)
    
    # Read the input JSON file
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Take the first n samples
    if len(data) > args.sample_size:
        sampled_data = data[:args.sample_size]
        print(f"Selected the first {args.sample_size} out of {len(data)} items")
    else:
        sampled_data = data
        print(f"Using all {len(data)} items (requested {args.sample_size})")
    
    # Extract videos, questions, and references
    videos = [item["video"] for item in sampled_data]
    questions = [item["question"] for item in sampled_data]
    references = [item.get("answer", item.get("reference", "")) for item in sampled_data]
    
    # Process each model separately
    total_models = len(args.models)
    for model_idx, model_spec in enumerate(args.models, 1):
        print(f"\n{'='*60}")
        print(f"Processing Model {model_idx}/{total_models}: {model_spec}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Process all videos with current model
        results = process_videos_with_model(model_spec, videos, questions)
        
        # Add reference captions to results for completeness
        for i, (result, reference) in enumerate(zip(results, references)):
            result["reference"] = reference
        
        model_name, checkpoint_name = parse_model_spec(model_spec)
        
        # Create metadata
        metadata = {
            "model_name": model_name,
            "checkpoint": checkpoint_name,
            "model_spec": model_spec,
            "input_file": input_path,
            "sample_size": args.sample_size,
            "generation_timestamp": datetime.now().isoformat(),
            "method_type": "Caption_Generation"
        }
        
        # Generate output filename and save
        output_filename = generate_output_filename(model_spec)
        output_path = os.path.join(args.output_dir, output_filename)
        
        save_captions(results, output_path, metadata)
        
        end_time = time.time()
        print(f"Completed model {model_spec} in {end_time - start_time:.2f} seconds")
    
    print(f"\nProcessed {len(videos)} videos with {total_models} models.")

if __name__ == "__main__":
    main()