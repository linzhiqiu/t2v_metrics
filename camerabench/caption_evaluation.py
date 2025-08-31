#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np
from collections import Counter
import string
from datetime import datetime
import openai
import time
import glob
from typing import List, Dict, Any, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def find_caption_files(score_dir: str) -> List[str]:
    """
    Auto-discover caption result files in the specified directory.
    
    Args:
        score_dir: Directory to search for caption files
        
    Returns:
        List of caption result file paths
    """
    if not os.path.exists(score_dir):
        print(f"Warning: Directory {score_dir} does not exist")
        return []
    
    # Look for files with pattern: caption_results_*.json
    pattern = os.path.join(score_dir, "caption_results_*.json")
    files = glob.glob(pattern)
    
    if files:
        print(f"Auto-discovered {len(files)} caption result files:")
        for f in sorted(files):
            print(f"  {os.path.basename(f)}")
    else:
        print(f"No caption result files found with pattern: caption_results_*.json")
    
    return sorted(files)


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
        return None  # Return None if no API key
    
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


def evaluate_caption_file(file_path: str, api_key: str = None) -> Dict[str, Any]:
    """
    Evaluate captions from a single result file.
    
    Args:
        file_path: Path to the caption result file
        api_key: OpenAI API key for GPT-4o judge
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load the caption results
    data = load_json_file(file_path)
    
    if not data or 'captions' not in data:
        print(f"Error: Invalid or missing caption data in {file_path}")
        return {}
    
    captions = data['captions']
    metadata = data.get('metadata', {})
    
    model_name = metadata.get('model_name', 'unknown')
    checkpoint = metadata.get('checkpoint', '')
    
    print(f"Evaluating {len(captions)} captions from model: {model_name}")
    if checkpoint:
        print(f"  Checkpoint: {checkpoint}")
    
    # Calculate statistics from the captions data
    total_samples = len(captions)
    failed_samples = sum(1 for caption in captions if caption.get('error') is not None)
    successful_samples = total_samples - failed_samples
    
    print(f"  Total samples: {total_samples}")
    print(f"  Successful samples: {successful_samples}")
    print(f"  Failed samples: {failed_samples}")
    
    # Initialize score lists
    spice_scores = []
    cider_scores = []
    bleu2_scores = []
    rouge_l_scores = []
    meteor_scores = []
    gen_match_scores = []
    
    # Process each caption
    valid_samples = 0
    for item in captions:
        reference = item.get("reference_answer", "")
        candidate = item.get("generated_caption", "")
        error = item.get("error")
        
        # Skip items with errors or missing data
        if error or not reference or not candidate:
            continue
            
        valid_samples += 1
        
        # Calculate metrics
        spice = calculate_spice_score(reference, candidate)
        spice_scores.append(spice)
        
        cider = calculate_cider_score(reference, candidate)
        cider_scores.append(cider)
        
        bleu2 = calculate_bleu2_score(reference, candidate)
        bleu2_scores.append(bleu2)
        
        rouge_l = calculate_rouge_l_score(reference, candidate)
        rouge_l_scores.append(rouge_l)
        
        meteor = calculate_meteor_score(reference, candidate)
        meteor_scores.append(meteor)
        
        # Calculate generative match if API key provided
        if api_key:
            gen_match = calculate_generative_match(reference, candidate, api_key)
            if gen_match is not None:
                gen_match_scores.append(gen_match)
    
    # Calculate averages
    results = {
        "model": model_name,
        "checkpoint": checkpoint,
        "file_path": file_path,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "spice": float(np.mean(spice_scores)) if spice_scores else 0.0,
        "cider": float(np.mean(cider_scores)) if cider_scores else 0.0,
        "bleu2": float(np.mean(bleu2_scores)) if bleu2_scores else 0.0,
        "rouge_l": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        "meteor": float(np.mean(meteor_scores)) if meteor_scores else 0.0,
        "gen_match": float(np.mean(gen_match_scores)) if gen_match_scores else None
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate caption generation results")
    
    # Input arguments
    parser.add_argument("files", nargs="*", help="Specific caption result files to evaluate")
    parser.add_argument("--score_dir", type=str, help="Directory to auto-discover caption result files")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, help="Output JSON file path")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Output directory for auto-generated filenames")
    
    # GPT-4o judge arguments
    parser.add_argument("--api_key", type=str, help="OpenAI API key for GPT-4o judge")
    parser.add_argument("--no_gpt", action="store_true", help="Skip GPT-4o judge evaluation")
    
    args = parser.parse_args()
    
    # Check for required NLTK data
    try:
        nltk.data.find('punkt')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
    
    # Determine which files to evaluate
    files_to_evaluate = []
    
    if args.files:
        # Use explicitly provided files
        files_to_evaluate = args.files
        print(f"Evaluating {len(files_to_evaluate)} explicitly provided files")
    elif args.score_dir:
        # Auto-discover files
        files_to_evaluate = find_caption_files(args.score_dir)
        if not files_to_evaluate:
            print("No caption result files found for evaluation")
            return
    else:
        print("Error: Please provide either specific files or --score_dir for auto-discovery")
        return
    
    # Get API key
    api_key = None
    if not args.no_gpt:
        api_key = get_openai_api_key(args.api_key)
        if api_key:
            print("Using OpenAI API key for GPT-4o judge evaluation")
        else:
            print("Warning: No OpenAI API key found. GPT-4o judge evaluation will be skipped.")
    else:
        print("GPT-4o judge evaluation disabled via --no_gpt flag")
    
    # Evaluate each file
    all_results = []
    
    for file_path in files_to_evaluate:
        print(f"\n{'='*50}")
        print(f"Evaluating: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        
        results = evaluate_caption_file(file_path, api_key)
        if results:
            all_results.append(results)
        else:
            print(f"Skipping {file_path} due to evaluation errors")
    
    if not all_results:
        print("No results to save. Exiting.")
        return
    
    # Print summary results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"\nModel: {result['model']}")
        if result['checkpoint']:
            print(f"  Checkpoint: {result['checkpoint']}")
        print(f"  Valid samples: {result['valid_samples']}/{result['total_samples']}")
        print(f"  SPICE: {result['spice']:.4f}")
        print(f"  CIDEr: {result['cider']:.4f}")
        print(f"  BLEU-2: {result['bleu2']:.4f}")
        print(f"  ROUGE-L: {result['rouge_l']:.4f}")
        print(f"  METEOR: {result['meteor']:.4f}")
        if result['gen_match'] is not None:
            print(f"  GPT-4o Judge: {result['gen_match']:.4f}")
    
    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        # Auto-generate filename
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_count = len(all_results)
        output_file = os.path.join(args.output_dir, f"caption_evaluation_{model_count}models_{timestamp}.json")
    
    # Save results to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluated_files": len(files_to_evaluate),
            "total_models": len(all_results),
            "gpt_judge_enabled": api_key is not None and not args.no_gpt,
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved evaluation results to: {output_file}")


if __name__ == "__main__":
    main()