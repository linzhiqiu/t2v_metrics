#!/usr/bin/env python3
import json
import os
import argparse
import t2v_metrics
from tqdm import tqdm
import time

def process_videos_with_model(model_name, videos, questions):
    """
    Process all videos with a single model instance.
    
    Args:
        model_name: Name of the model to use
        videos: List of video paths
        questions: List of questions corresponding to videos
        
    Returns:
        List of generated captions
    """
    print(f"\nLoading model: {model_name}")
    try:
        # Initialize the model once
        if 'gemini' in model_name:
            score_model = t2v_metrics.get_score_model(model=model_name, api_key='api_key')
        elif 'gpt' in model_name:
            score_model = t2v_metrics.get_score_model(model=model_name, api_key='api_key')
        else:
            score_model = t2v_metrics.VQAScore(model=model_name)
        
        captions = []
        # Process all videos with this model
        for i, (video, question) in enumerate(tqdm(zip(videos, questions), total=len(videos), desc=f"Processing with {model_name}")):
            try:
                # Generate the caption
                response = score_model.model.generate(images=[video], texts=[question])
                caption = response[0] if isinstance(response, list) else response
                captions.append(caption)
            except Exception as e:
                error_msg = f"Error processing video {i} with {model_name}: {str(e)}"
                print(error_msg)
                captions.append(f"Error: {str(e)}")
                
        return captions
    except Exception as e:
        print(f"Fatal error with model {model_name}: {str(e)}")
        # Return error messages for all videos if model initialization fails
        return [f"Error loading model {model_name}: {str(e)}"] * len(videos)

def main():
    parser = argparse.ArgumentParser(description="Generate captions using different models")
    parser.add_argument("--input", type=str, default="~/test_caption.json", help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="model_captions.json", help="Path to output JSON file")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of items to sample")
    parser.add_argument("--models", type=str, nargs="+", default=["gpt-4o", "gemini-2.0", "qwen2.5-vl-cam15000"], 
                        help="List of model names to use")
    args = parser.parse_args()
    
    # Expand the tilde in the path
    input_path = os.path.expanduser(args.input)
    
    # Read the input JSON file
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Take the first n samples instead of random samples
    if len(data) > args.sample_size:
        sampled_data = data[:args.sample_size]
        print(f"Selected the first {args.sample_size} out of {len(data)} items")
    else:
        sampled_data = data
        print(f"Using all {len(data)} items (requested {args.sample_size})")
    
    # Extract videos and questions
    videos = [item["video"] for item in sampled_data]
    questions = [item["question"] for item in sampled_data]
    references = [item["answer"] for item in sampled_data]
    
    # Initialize results structure
    results = []
    for i, (video, question, reference) in enumerate(zip(videos, questions, references)):
        results.append({
            "video": video,
            "question": question,
            "reference": reference
        })
    
    # Process each model one at a time
    for model_name in args.models:
        start_time = time.time()
        
        # Process all videos with current model
        captions = process_videos_with_model(model_name, videos, questions)
        
        # Add captions to results
        for i, caption in enumerate(captions):
            results[i][model_name] = caption
            
        end_time = time.time()
        print(f"Completed model {model_name} in {end_time - start_time:.2f} seconds")
        
        # Save intermediate results after each model is processed
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Intermediate results saved to {args.output}")
    
    # Final save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} videos with {len(args.models)} models. Final results saved to {args.output}")

if __name__ == "__main__":
    main()