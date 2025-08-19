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
        List of generated captions (each item contains top 5 sequences)
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
                # Generate top 5 sequences
                # Check if the model supports generate parameters
                try:
                    # Try with sampling parameters for top-k generation
                    response = score_model.model.generate(
                        images=[video], 
                        texts=[question],
                        num_return_sequences=5,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7,
                        max_new_tokens=256
                    )
                except Exception as generation_error:
                    # Fallback: try multiple calls with different temperatures
                    print(f"Standard top-k generation failed for {model_name}, trying temperature sampling...")
                    response = []
                    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different temperatures for diversity
                    
                    for temp in temperatures:
                        try:
                            temp_response = score_model.model.generate(
                                images=[video], 
                                texts=[question],
                                temperature=temp,
                                do_sample=True,
                                max_new_tokens=256
                            )
                            temp_caption = temp_response[0] if isinstance(temp_response, list) else temp_response
                            response.append(temp_caption)
                        except Exception as temp_error:
                            print(f"Temperature {temp} failed: {temp_error}")
                            response.append(f"Error with temp {temp}: {str(temp_error)}")
                
                # Ensure we have a list of captions
                if not isinstance(response, list):
                    response = [response]
                
                # Pad with duplicates if we have fewer than 5
                while len(response) < 5:
                    response.append(response[0] if response else "Error: No response generated")
                
                # Take only top 5
                top_5_captions = response[:5]
                captions.append(top_5_captions)
                
            except Exception as e:
                error_msg = f"Error processing video {i} with {model_name}: {str(e)}"
                print(error_msg)
                # Return 5 error messages
                captions.append([f"Error: {str(e)}"] * 5)
                
        return captions
    except Exception as e:
        print(f"Fatal error with model {model_name}: {str(e)}")
        # Return error messages for all videos if model initialization fails
        return [[f"Error loading model {model_name}: {str(e)}"] * 5] * len(videos)

def main():
    parser = argparse.ArgumentParser(description="Generate top-5 captions using different models")
    parser.add_argument("--input", type=str, default="~/test_caption.json", help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="cambench_qwen_7b_sft_dpo_5000_top5.json", help="Path to output JSON file")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of items to sample")
    parser.add_argument("--models", type=str, nargs="+", default=['qwen2.5-vl-cambench-sft-dpo-5000'], 
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
        captions_list = process_videos_with_model(model_name, videos, questions)
        
        # Add captions to results
        for i, top_5_captions in enumerate(captions_list):
            # Store as both individual sequences and as a list
            results[i][f"{model_name}_top5"] = top_5_captions
            results[i][f"{model_name}_best"] = top_5_captions[0]  # Keep best for compatibility
            
            # Also store individual sequences for easier access
            for j, caption in enumerate(top_5_captions):
                results[i][f"{model_name}_seq_{j+1}"] = caption
            
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
    print("Each result contains:")
    print("  - {model_name}_top5: List of top 5 sequences")
    print("  - {model_name}_best: Best sequence (same as seq_1)")
    print("  - {model_name}_seq_1 to {model_name}_seq_5: Individual sequences")

if __name__ == "__main__":
    main()