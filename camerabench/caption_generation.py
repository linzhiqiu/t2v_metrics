#!/usr/bin/env python3
import t2v_metrics
import json
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any, Tuple


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


def load_caption_data(data_path: str) -> List[Dict[str, Any]]:
    """Load caption data from JSON file."""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading caption data from {data_path}: {e}")
        return []


def generate_captions_for_model(model_name: str, checkpoint: str, data: List[Dict[str, Any]], 
                               video_dir: str, sample_size: int = None) -> Dict[str, Any]:
    """
    Generate captions using a single model.
    
    Args:
        model_name: Name of the model to use
        checkpoint: Checkpoint path (can be None)
        data: List of caption data items
        video_dir: Base directory for video files
        sample_size: Number of samples to process (None for all)
        
    Returns:
        Dictionary with metadata and caption results
    """
    print(f"\nLoading model: {model_name}")
    if checkpoint:
        print(f"Using checkpoint: {checkpoint}")
    print(f"Using video directory: {video_dir}")
    
    # Sample data if requested
    if sample_size and len(data) > sample_size:
        sampled_data = data[:sample_size]
        print(f"Using first {sample_size} out of {len(data)} samples")
    else:
        sampled_data = data
        print(f"Using all {len(data)} samples")
    
    try:
        # Initialize the model
        if 'gemini' in model_name:
            score_model = t2v_metrics.get_score_model(model=model_name, api_key='api_key')
        elif 'gpt' in model_name:
            score_model = t2v_metrics.get_score_model(model=model_name, api_key='api_key')
        else:
            # For other models, pass checkpoint parameter if provided
            if checkpoint:
                score_model = t2v_metrics.VQAScore(model=model_name, checkpoint=checkpoint)
            else:
                score_model = t2v_metrics.VQAScore(model=model_name)
        
        # Generate captions for all samples
        captions = []
        
        for i, item in enumerate(tqdm(sampled_data, desc=f"Generating captions with {model_name}")):
            video_path = item.get("video", "")
            
            # Concatenate video_dir with video_path
            full_video_path = os.path.join(video_dir, video_path) if video_path else ""
            
            question = item.get("question", "")
            reference_answer = item.get("answer", item.get("reference", ""))
            
            try:
                # Generate the caption
                response = score_model.model.generate(images=[full_video_path], texts=[question])
                caption = response[0] if isinstance(response, list) else response
                
                captions.append({
                    "sample_id": str(i),
                    "video_path": video_path,  # Keep original path in output
                    "full_video_path": full_video_path,  # Store full path for reference
                    "question": question,
                    "reference_answer": reference_answer,
                    "method": model_name,
                    "generated_caption": caption,
                    "error": None
                })
                
            except Exception as e:
                error_msg = f"Error processing sample {i}: {str(e)}"
                print(error_msg)
                
                captions.append({
                    "sample_id": str(i),
                    "video_path": video_path,
                    "full_video_path": full_video_path,
                    "question": question,
                    "reference_answer": reference_answer,
                    "method": model_name,
                    "generated_caption": "",
                    "error": str(e)
                })
        
        # Prepare metadata
        metadata = {
            "method_type": "VLM_Caption_Generation",
            "model_name": model_name,
            "checkpoint": checkpoint,
            "video_dir": video_dir,
            "generation_timestamp": datetime.now().isoformat()
        }
        
        return {
            "metadata": metadata,
            "captions": captions
        }
        
    except Exception as e:
        print(f"Fatal error with model {model_name}: {str(e)}")
        
        # Return error structure
        error_captions = []
        for i, item in enumerate(sampled_data):
            video_path = item.get("video", "")
            full_video_path = os.path.join(video_dir, video_path) if video_path else ""
            
            error_captions.append({
                "sample_id": str(i),
                "video_path": video_path,
                "full_video_path": full_video_path,
                "question": item.get("question", ""),
                "reference_answer": item.get("answer", item.get("reference", "")),
                "method": model_name,
                "generated_caption": "",
                "error": f"Model loading error: {str(e)}"
            })
        
        metadata = {
            "method_type": "VLM_Caption_Generation", 
            "model_name": model_name,
            "checkpoint": checkpoint,
            "video_dir": video_dir,
            "generation_timestamp": datetime.now().isoformat()
        }
        
        return {
            "metadata": metadata,
            "captions": error_captions
        }


def create_output_filename(model_name: str, checkpoint: str, output_dir: str) -> str:
    """Create standardized output filename for caption results."""
    # Clean model name for filename
    clean_model_name = model_name.replace("/", "_").replace(":", "_")
    
    # Create base filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if checkpoint:
        # Clean checkpoint name
        clean_checkpoint = os.path.basename(checkpoint).replace("/", "_").replace(":", "_")
        filename = f"caption_results_{clean_model_name}_{clean_checkpoint}_{timestamp}.json"
    else:
        filename = f"caption_results_{clean_model_name}_{timestamp}.json"
    
    return os.path.join(output_dir, filename)


def main():
    parser = argparse.ArgumentParser(description="Generate captions using vision-language models")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to use for caption generation")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing caption_data.json")
    parser.add_argument("--output_dir", type=str, default="scores",
                        help="Directory to save caption generation results")
    
    # Optional arguments
    parser.add_argument("--video_dir", type=str, default="data/videos",
                        help="Base directory for video files (will be concatenated with video paths)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (optional)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to process (default: all)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine caption data file path
    caption_data_path = os.path.join(args.data_dir, "caption_data.json")
    
    # Load caption data
    print(f"Loading caption data from: {caption_data_path}")
    data = load_caption_data(caption_data_path)
    
    if not data:
        print("Error: No caption data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data)} caption samples")
    
    # Generate captions
    print(f"\nGenerating captions with model: {args.model}")
    if args.checkpoint:
        print(f"Using checkpoint: {args.checkpoint}")
    if args.video_dir:
        print(f"Using video directory: {args.video_dir}")
    if args.sample_size:
        print(f"Processing {args.sample_size} samples")
    
    start_time = time.time()
    
    results = generate_captions_for_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        data=data,
        video_dir=args.video_dir,
        sample_size=args.sample_size
    )
    
    end_time = time.time()
    
    # Create output filename
    output_file = create_output_filename(args.model, args.checkpoint, args.output_dir)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Calculate statistics for display
    total_samples = len(results['captions'])
    successful_samples = sum(1 for caption in results['captions'] if caption['error'] is None)
    failed_samples = total_samples - successful_samples
    
    print(f"\nCaption generation completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_file}")
    print(f"Successfully generated captions for {successful_samples}/{total_samples} samples")
    
    if failed_samples > 0:
        print(f"Failed samples: {failed_samples}")


if __name__ == "__main__":
    main()