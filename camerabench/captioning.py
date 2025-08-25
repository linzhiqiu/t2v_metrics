#!/usr/bin/env python3
import t2v_metrics
import json
import os
import argparse
import numpy as np
from collections import Counter
import string
from tqdm import tqdm
import pandas as pd
import openai
import time
import re
from typing import List, Dict, Any, Tuple, Set
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from rouge_score import rouge_scorer
import nltk


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
        List of generated captions
    """
    model_name, checkpoint_name = parse_model_spec(model_spec)
    
    print(f"\nLoading model: {model_name}")
    if checkpoint_name:
        print(f"Using checkpoint: {checkpoint_name}")
    
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


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def preprocess_text(text):
    """Preprocess text for evaluation"""
    # Handle None or empty string
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into words
    words = text.split()
    return words


def calculate_spice_score(reference, candidate):
    """Simplified SPICE calculation"""
    # Handle None or empty values
    if not reference or not candidate:
        return 0.0
    
    # Preprocess texts
    ref_words = set(preprocess_text(reference))
    cand_words = set(preprocess_text(candidate))
    
    # Calculate precision and recall
    if len(cand_words) == 0:
        return 0.0
    
    intersection = ref_words.intersection(cand_words)
    precision = len(intersection) / len(cand_words)
    recall = len(intersection) / len(ref_words) if len(ref_words) > 0 else 0
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def calculate_cider_score(reference, candidate):
    """Simplified CIDEr calculation"""
    # Handle None or empty values
    if not reference or not candidate:
        return 0.0
    
    # Preprocess texts
    ref_words = preprocess_text(reference)
    cand_words = preprocess_text(candidate)
    
    # Calculate word counts
    ref_counts = Counter(ref_words)
    cand_counts = Counter(cand_words)
    
    # Calculate cosine similarity
    all_words = set(ref_counts.keys()).union(set(cand_counts.keys()))
    
    if not all_words:
        return 0.0
    
    dot_product = sum(ref_counts[word] * cand_counts[word] for word in all_words)
    ref_magnitude = np.sqrt(sum(ref_counts[word] ** 2 for word in ref_counts))
    cand_magnitude = np.sqrt(sum(cand_counts[word] ** 2 for word in cand_counts))
    
    if ref_magnitude == 0 or cand_magnitude == 0:
        return 0.0
    
    similarity = dot_product / (ref_magnitude * cand_magnitude)
    
    return similarity


def calculate_bleu2_score(reference, candidate):
    """
    Calculate BLEU-2 score (up to bigrams)
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        BLEU-2 score
    """
    if not reference or not candidate:
        return 0.0
    
    # Tokenize texts
    ref_tokens = preprocess_text(reference)
    cand_tokens = preprocess_text(candidate)
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    # Use smoothing to avoid zero scores when there are no matches
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU-2 score (weights for unigrams and bigrams only)
    weights = (0.5, 0.5)  # Equal weights for unigrams and bigrams
    
    try:
        score = sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"Error calculating BLEU-2: {e}")
        return 0.0


def calculate_rouge_l_score(reference, candidate):
    """
    Calculate ROUGE-L score (longest common subsequence)
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        ROUGE-L F1 score
    """
    if not reference or not candidate:
        return 0.0
    
    try:
        # Initialize Rouge scorer with RougeL
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Calculate scores
        scores = scorer.score(reference, candidate)
        
        # Return F1 score
        return scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Error calculating ROUGE-L: {e}")
        return 0.0


def calculate_meteor_score(reference, candidate):
    """
    Calculate METEOR score
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        METEOR score
    """
    if not reference or not candidate:
        return 0.0

    try:
        # Ensure nltk data is available
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
        # Tokenize texts
        ref_tokens = preprocess_text(reference)
        cand_tokens = preprocess_text(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
            
        # Create sets of unigrams, bigrams, and trigrams
        ref_unigrams = set(ref_tokens)
        cand_unigrams = set(cand_tokens)
        
        ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:])) if len(ref_tokens) > 1 else set()
        cand_bigrams = set(zip(cand_tokens[:-1], cand_tokens[1:])) if len(cand_tokens) > 1 else set()
        
        # Calculate precision and recall for unigrams
        unigram_matches = len(ref_unigrams.intersection(cand_unigrams))
        unigram_precision = unigram_matches / len(cand_unigrams) if cand_unigrams else 0
        unigram_recall = unigram_matches / len(ref_unigrams) if ref_unigrams else 0
        
        # Calculate precision and recall for bigrams
        bigram_matches = len(ref_bigrams.intersection(cand_bigrams))
        bigram_precision = bigram_matches / len(cand_bigrams) if cand_bigrams else 0
        bigram_recall = bigram_matches / len(ref_bigrams) if ref_bigrams else 0
        
        # Calculate weighted precision and recall (unigrams weighted more)
        precision = (0.8 * unigram_precision + 0.2 * bigram_precision)
        recall = (0.8 * unigram_recall + 0.2 * bigram_recall)
        
        # Calculate METEOR-like score (with simplified components)
        if precision + recall == 0:
            return 0.0
            
        # Apply harmonic mean with recall weighted higher (as in METEOR)
        meteor_score = (10 * precision * recall) / (recall + 9 * precision)
        
        return meteor_score
    except Exception as e:
        print(f"Error calculating METEOR: {e}")
        return 0.0


def get_openai_api_key(provided_key=None):
    """
    Get OpenAI API key from argument or environment variable.
    
    Args:
        provided_key: API key provided as command line argument
        
    Returns:
        API key string or None if not found
    """
    if provided_key:
        return provided_key
    
    # Try environment variable
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    
    return None


def calculate_generative_match(reference, candidate, api_key=None, retries=3, delay=2):
    """
    Calculate generative match score using GPT-4o as judge.
    
    Args:
        reference: Reference caption
        candidate: Model-generated caption
        api_key: OpenAI API key
        retries: Number of retries if API call fails
        delay: Delay between retries in seconds
        
    Returns:
        Logit probability for "Yes" response
    """
    if not reference or not candidate:
        return 0.0
    
    # Set up OpenAI API
    if api_key:
        openai.api_key = api_key
    else:
        print("Warning: No OpenAI API key provided. Returning placeholder score.")
        return 0.5  # Return placeholder score
    
    prompt = f"Reference caption: '{reference}'\nCandidate caption: '{candidate}'\n\nDoes the candidate caption match the reference caption? Answer Yes or No."
    
    for attempt in range(retries):
        try:
            # Call GPT-4o API
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
                logprobs=True,
                top_logprobs=5
            )
            
            # Extract response and logprobs
            content = response.choices[0].message.content.strip().lower()
            
            # Check if the answer is clearly yes or no
            if content.startswith("yes"):
                return 1.0
            elif content.startswith("no"):
                return 0.0
            
            # If we can't determine from the content, try to get the logprobs
            logprobs = response.choices[0].logprobs.content[0].top_logprobs
            
            # Look for "yes" in top logprobs
            for token_info in logprobs:
                if token_info.token.strip().lower() == "yes":
                    return np.exp(token_info.logprob)  # Convert log probability to probability
            
            # If "yes" not found in top logprobs, return low probability
            return 0.1
            
        except Exception as e:
            print(f"Error calling OpenAI API (attempt {attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries exceeded. Using fallback score.")
                return 0.5  # Fallback score
    
    return 0.5  # Should never reach here but just in case


def evaluate_models_from_combined(combined_data, api_key=None):
    """
    Evaluate models from a combined JSON file.
    
    Args:
        combined_data: List of dictionaries with combined model outputs
        api_key: OpenAI API key for GPT-4o judge
        
    Returns:
        Dictionary mapping model names to their evaluation results
    """
    # Extract model names (fields that are not common fields)
    common_fields = {"video", "question", "reference", "answer"}
    
    if len(combined_data) == 0:
        print("Error: No data found in the input")
        return {}
    
    all_fields = set(combined_data[0].keys())
    model_names = list(all_fields - common_fields)
    
    if not model_names:
        print("Error: No model outputs found in the data")
        return {}
    
    print(f"Found {len(model_names)} models: {', '.join(model_names)}")
    
    # Prepare data structure for results
    results = {}
    
    # For each model, calculate metrics
    for model_name in model_names:
        print(f"\nEvaluating model: {model_name}")
        
        spice_scores = []
        cider_scores = []
        bleu2_scores = []
        rouge_l_scores = []
        meteor_scores = []
        gen_match_scores = []
        
        # Process each item
        for item in tqdm(combined_data, desc=f"Processing {model_name}"):
            reference = item.get("reference", "") or item.get("answer", "")
            
            # Skip if this model doesn't have a caption for this item
            if model_name not in item:
                continue
                
            candidate = item.get(model_name, "")
            
            # Skip items with missing data
            if not reference or not candidate:
                continue
                
            # Calculate SPICE score
            spice = calculate_spice_score(reference, candidate)
            spice_scores.append(spice)
            
            # Calculate CIDEr score
            cider = calculate_cider_score(reference, candidate)
            cider_scores.append(cider)
            
            # Calculate BLEU-2 score
            bleu2 = calculate_bleu2_score(reference, candidate)
            bleu2_scores.append(bleu2)
            
            # Calculate ROUGE-L score
            rouge_l = calculate_rouge_l_score(reference, candidate)
            rouge_l_scores.append(rouge_l)
            
            # Calculate METEOR score
            meteor = calculate_meteor_score(reference, candidate)
            meteor_scores.append(meteor)
            
            # Calculate generative match score
            if api_key:
                gen_match = calculate_generative_match(reference, candidate, api_key)
                gen_match_scores.append(gen_match)
        
        # Calculate averages
        avg_spice = float(np.mean(spice_scores)) if spice_scores else 0.0
        avg_cider = float(np.mean(cider_scores)) if cider_scores else 0.0
        avg_bleu2 = float(np.mean(bleu2_scores)) if bleu2_scores else 0.0
        avg_rouge_l = float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0
        avg_meteor = float(np.mean(meteor_scores)) if meteor_scores else 0.0
        avg_gen_match = float(np.mean(gen_match_scores)) if gen_match_scores else None
        
        # Store results
        results[model_name] = {
            "model": model_name,
            "spice": avg_spice,
            "cider": avg_cider,
            "bleu2": avg_bleu2,
            "rouge_l": avg_rouge_l,
            "meteor": avg_meteor,
            "gen_match": avg_gen_match,
            "count": len(spice_scores)
        }
    
    return results


def generate_captions(args):
    """Generate captions using specified models"""
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
    references = [item.get("answer", item.get("reference", "")) for item in sampled_data]
    
    # Initialize results structure
    results = []
    for i, (video, question, reference) in enumerate(zip(videos, questions, references)):
        results.append({
            "video": video,
            "question": question,
            "reference": reference
        })
    
    # Process videos with each model
    total_models = len(args.models)
    for model_idx, model_spec in enumerate(args.models, 1):
        print(f"\n{'='*50}")
        print(f"Processing Model {model_idx}/{total_models}: {model_spec}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Process all videos with the model
        captions = process_videos_with_model(model_spec, videos, questions)
        
        # Create model key (use the full model_spec as the key)
        model_key = model_spec
        
        # Add captions to results
        for i, caption in enumerate(captions):
            results[i][model_key] = caption
            
        end_time = time.time()
        print(f"Completed model {model_key} in {end_time - start_time:.2f} seconds")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    print(f"Processed {len(results)} videos with {total_models} models.")
    return results


def evaluate_captions(args, data=None):
    """Evaluate captions from a combined JSON file"""
    # Check for required NLTK data
    try:
        nltk.data.find('punkt')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
    
    # Load data if not provided
    if data is None:
        data = load_json_file(args.input)
        if not data:
            print("Error: Could not load data from input file")
            return None
    
    # Get API key from arguments or environment
    api_key = get_openai_api_key(getattr(args, 'api_key', None))
    
    if not getattr(args, 'no_gpt', False) and api_key:
        print(f"Using OpenAI API key for GPT-4o judge evaluation")
    elif not getattr(args, 'no_gpt', False):
        print("Warning: No OpenAI API key found. GPT-4o judge evaluation will be skipped.")
        api_key = None
    else:
        print("GPT-4o judge evaluation disabled via --no_gpt flag")
        api_key = None
    
    # Evaluate models
    evaluation_results = evaluate_models_from_combined(data, api_key)
    
    # Convert results to list for easier serialization
    results_list = list(evaluation_results.values())
    
    # Print results
    print("\nEvaluation Results:")
    for model_name, result in evaluation_results.items():
        print(f"Model: {model_name}")
        print(f"  SPICE: {result['spice']:.4f}")
        print(f"  CIDEr: {result['cider']:.4f}")
        print(f"  BLEU-2: {result['bleu2']:.4f}")
        print(f"  ROUGE-L: {result['rouge_l']:.4f}")
        print(f"  METEOR: {result['meteor']:.4f}")
        if result['gen_match'] is not None:
            print(f"  GPT-4o Judge Score: {result['gen_match']:.4f}")
        print(f"  Sample count: {result['count']}")
    
    # Save results to JSON
    eval_output = getattr(args, 'eval_output', 'evaluation_results.json')
    with open(eval_output, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\nSaved evaluation results to {eval_output}")
    
    # Export to Excel
    try:
        # Create DataFrame for evaluation results
        results_df = pd.DataFrame(results_list)
        
        # Create a model comparison DataFrame sorted by generative match score
        if any(r['gen_match'] is not None for r in results_list):
            sort_key = 'gen_match'
        else:
            sort_key = 'cider'
        
        comparison_df = results_df.sort_values(by=sort_key, ascending=False)
        
        # Export to Excel with multiple sheets
        excel_output = getattr(args, 'excel_output', 'evaluation_results.xlsx')
        with pd.ExcelWriter(excel_output) as writer:
            results_df.to_excel(writer, sheet_name='All Results', index=False)
            comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
            
            # Add a sheet with detailed item-level results for further analysis
            if len(data) <= 1000:  # Only if the data is not too large
                detailed_results = []
                
                for item in data:
                    video = item.get("video", "")
                    question = item.get("question", "")
                    reference = item.get("reference", "") or item.get("answer", "")
                    
                    for model_name in evaluation_results.keys():
                        if model_name in item:
                            candidate = item.get(model_name, "")
                            
                            spice = calculate_spice_score(reference, candidate)
                            cider = calculate_cider_score(reference, candidate)
                            bleu2 = calculate_bleu2_score(reference, candidate)
                            rouge_l = calculate_rouge_l_score(reference, candidate)
                            meteor = calculate_meteor_score(reference, candidate)
                            
                            detailed_results.append({
                                "video": video,
                                "question": question,
                                "model": model_name,
                                "reference": reference,
                                "candidate": candidate,
                                "spice": spice,
                                "cider": cider,
                                "bleu2": bleu2,
                                "rouge_l": rouge_l,
                                "meteor": meteor
                            })
                
                # Create DataFrame and sort by model and video
                detailed_df = pd.DataFrame(detailed_results)
                detailed_df = detailed_df.sort_values(by=["model", "video"])
                
                # Export to the Excel file
                detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        print(f"Exported results to Excel file: {excel_output}")
    except ImportError:
        print("Warning: pandas is required for Excel export. Install with 'pip install pandas openpyxl'.")
    
    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Generate captions and/or evaluate model outputs")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["generate", "evaluate", "both"], default="both",
                        help="Mode: generate captions, evaluate existing results, or both")
    
    # Input/Output arguments
    parser.add_argument("--input", type=str, default="~/test_caption.json", 
                        help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="caption_results.json", 
                        help="Path to output JSON file for generated captions")
    
    # Caption generation arguments - multiple models approach
    parser.add_argument("--models", type=str, nargs='+', 
                        default=["qwen2.5-vl-7b"],
                        help="List of model names to use for caption generation. Format: model_name or model_name:checkpoint")
    parser.add_argument("--sample-size", type=int, default=100, 
                        help="Number of items to sample for caption generation")
    
    # Backwards compatibility - single model arguments (deprecated)
    parser.add_argument("--model", type=str, 
                        help="Single model name (deprecated, use --models instead)")
    parser.add_argument("--checkpoint", type=str, 
                        help="Single model checkpoint (deprecated, use --models with model:checkpoint format)")
    
    # Evaluation arguments  
    parser.add_argument("--eval_output", type=str, default="evaluation_results.json",
                        help="Path to evaluation results JSON file")
    parser.add_argument("--excel_output", type=str, default="evaluation_results.xlsx",
                        help="Path to evaluation results Excel file")
    parser.add_argument("--api_key", type=str, help="OpenAI API key for GPT-4o judge (also checks OPENAI_API_KEY env var)")
    parser.add_argument("--no_gpt", action="store_true", help="Skip GPT-4o judge evaluation")
    
    args = parser.parse_args()
    
    # Handle backwards compatibility for single model arguments
    if args.model and not hasattr(args, 'models_specified'):
        if args.checkpoint:
            single_model_spec = f"{args.model}:{args.checkpoint}"
        else:
            single_model_spec = args.model
        args.models = [single_model_spec]
        print(f"Using single model specification: {single_model_spec}")
    
    if args.mode == "generate":
        print("Mode: Caption Generation Only")
        print(f"Models to process: {', '.join(args.models)}")
        generate_captions(args)
        
    elif args.mode == "evaluate":
        print("Mode: Evaluation Only")
        evaluate_captions(args)
        
    elif args.mode == "both":
        print("Mode: Caption Generation + Evaluation")
        print(f"Models to process: {', '.join(args.models)}")
        
        # First generate captions
        results = generate_captions(args)
        
        # Then evaluate the generated captions
        print("\n" + "="*50)
        print("Starting Evaluation Phase...")
        print("="*50)
        evaluate_captions(args, data=results)


if __name__ == "__main__":
    main()