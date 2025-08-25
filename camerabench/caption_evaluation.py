#!/usr/bin/env python3
"""
Method-agnostic caption evaluator.
Takes standardized caption files and computes comprehensive evaluation metrics 
(SPICE, CIDEr, BLEU-2, ROUGE-L, METEOR, GPT-4o Judge).
This script works with any method that outputs captions in the expected format.
"""

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
from datetime import datetime
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

def load_caption_file(caption_file):
    """Load captions from a standardized caption file"""
    with open(caption_file, 'r') as f:
        data = json.load(f)
    
    return data

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

def evaluate_single_file(caption_file, api_key=None, use_gpt_judge=True):
    """Evaluate a single caption file"""
    print(f"\nEvaluating: {caption_file}")
    
    # Load caption data
    caption_data = load_caption_file(caption_file)
    
    # Extract metadata
    metadata = caption_data.get("metadata", {})
    method_name = metadata.get("method_type", "Unknown_Method")
    model_name = metadata.get("model_name", "Unknown_Model")
    model_spec = metadata.get("model_spec", model_name)
    
    print(f"Method: {method_name}")
    print(f"Model: {model_spec}")
    print(f"Total samples in file: {caption_data['total_samples']}")
    print(f"Successful samples: {caption_data['successful_samples']}")
    print(f"Failed samples: {caption_data['failed_samples']}")
    
    # Extract captions and references
    captions_list = caption_data["captions"]
    
    spice_scores = []
    cider_scores = []
    bleu2_scores = []
    rouge_l_scores = []
    meteor_scores = []
    gen_match_scores = []
    
    # Process each caption
    for item in tqdm(captions_list, desc="Evaluating captions"):
        if item["error"] is not None:
            continue  # Skip failed samples
            
        reference = item.get("reference", "")
        candidate = item.get("caption", "")
        
        # Skip items with missing data
        if not reference or not candidate:
            continue
            
        # Calculate all metrics
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
        
        # Calculate generative match score
        if use_gpt_judge and api_key:
            gen_match = calculate_generative_match(reference, candidate, api_key)
            gen_match_scores.append(gen_match)
    
    # Calculate metrics
    metrics = {
        "model_name": model_spec,
        "spice": float(np.mean(spice_scores)) if spice_scores else 0.0,
        "cider": float(np.mean(cider_scores)) if cider_scores else 0.0,
        "bleu2": float(np.mean(bleu2_scores)) if bleu2_scores else 0.0,
        "rouge_l": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        "meteor": float(np.mean(meteor_scores)) if meteor_scores else 0.0,
        "gen_match": float(np.mean(gen_match_scores)) if gen_match_scores else None,
        "evaluated_samples": len(spice_scores),
        "metadata": metadata
    }
    
    # Print results
    print(f"Evaluation Results:")
    print(f"  SPICE: {metrics['spice']:.4f}")
    print(f"  CIDEr: {metrics['cider']:.4f}")
    print(f"  BLEU-2: {metrics['bleu2']:.4f}")
    print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"  METEOR: {metrics['meteor']:.4f}")
    if metrics['gen_match'] is not None:
        print(f"  GPT-4o Judge Score: {metrics['gen_match']:.4f}")
    print(f"  Evaluated samples: {metrics['evaluated_samples']}")
    
    return model_spec, metrics

def save_evaluation_results(results, output_file):
    """Save evaluation results to JSON file"""
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_models": len(results),
        "results_by_model": results
    }
    
    # Compute overall statistics if multiple models
    if len(results) > 1:
        valid_results = [r for r in results.values() if r is not None]
        if valid_results:
            metrics_names = ['spice', 'cider', 'bleu2', 'rouge_l', 'meteor']
            overall_stats = {}
            
            for metric in metrics_names:
                values = [r[metric] for r in valid_results]
                overall_stats[f"mean_{metric}"] = float(np.mean(values))
                overall_stats[f"std_{metric}"] = float(np.std(values))
            
            # Handle gen_match separately since it can be None
            gen_match_scores = [r["gen_match"] for r in valid_results if r["gen_match"] is not None]
            if gen_match_scores:
                overall_stats["mean_gen_match"] = float(np.mean(gen_match_scores))
                overall_stats["std_gen_match"] = float(np.std(gen_match_scores))
            
            summary["overall_statistics"] = overall_stats
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_file}")

def export_to_excel(results, excel_file, detailed_results=None):
    """Export results to Excel file"""
    try:
        # Convert results to DataFrame
        results_list = list(results.values())
        results_df = pd.DataFrame(results_list)
        
        # Sort by best performing metric
        if 'gen_match' in results_df.columns and results_df['gen_match'].notna().any():
            sort_key = 'gen_match'
        else:
            sort_key = 'cider'
        
        sorted_df = results_df.sort_values(by=sort_key, ascending=False, na_last=True)
        
        # Export to Excel
        with pd.ExcelWriter(excel_file) as writer:
            results_df.to_excel(writer, sheet_name='All Results', index=False)
            sorted_df.to_excel(writer, sheet_name='Model Ranking', index=False)
            
            # Add detailed results if provided
            if detailed_results is not None:
                detailed_df = pd.DataFrame(detailed_results)
                detailed_df = detailed_df.sort_values(by=["model", "video"])
                detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        print(f"Exported results to Excel file: {excel_file}")
    except ImportError:
        print("Warning: pandas is required for Excel export. Install with 'pip install pandas openpyxl'.")

def main():
    parser = argparse.ArgumentParser(description='Method-agnostic caption evaluator')
    parser.add_argument('caption_files', nargs='+', 
                      help='Caption files to evaluate (JSON format from caption generation step)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file name for results (auto-generated if not provided)')
    parser.add_argument('--excel_file', type=str, default=None,
                      help='Excel file name for results (auto-generated if not provided)')
    parser.add_argument('--api_key', type=str, help='OpenAI API key for GPT-4o judge')
    parser.add_argument('--no_gpt', action='store_true', help='Skip GPT-4o judge evaluation')
    parser.add_argument('--detailed_excel', action='store_true', 
                      help='Include detailed per-sample results in Excel (for files with ≤1000 samples)')
    
    args = parser.parse_args()
    
    # Check for required NLTK data
    try:
        nltk.data.find('punkt')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"{'='*80}")
    print("METHOD-AGNOSTIC CAPTION EVALUATION")
    print(f"{'='*80}")
    print(f"Caption files to evaluate: {len(args.caption_files)}")
    
    # Determine if we should use GPT judge
    use_gpt_judge = not args.no_gpt
    api_key = get_openai_api_key(args.api_key) if use_gpt_judge else None
    
    if use_gpt_judge and not api_key:
        print("Warning: No OpenAI API key provided. GPT-4o judge evaluation will be skipped.")
        use_gpt_judge = False
    elif use_gpt_judge:
        print(f"Using GPT-4o judge evaluation")
    else:
        print("GPT-4o judge evaluation disabled")
    
    # Evaluate each file
    results = {}
    detailed_results = []
    
    for caption_file in args.caption_files:
        if not Path(caption_file).exists():
            print(f"Warning: Caption file not found: {caption_file}")
            continue
        
        model_spec, metrics = evaluate_single_file(caption_file, api_key, use_gpt_judge)
        
        if metrics:
            results[model_spec] = metrics
            
            # Collect detailed results if requested
            if args.detailed_excel:
                caption_data = load_caption_file(caption_file)
                if caption_data['total_samples'] <= 1000:
                    for item in caption_data["captions"]:
                        if item["error"] is None and item.get("reference") and item.get("caption"):
                            detailed_entry = {
                                "video": item["video"],
                                "question": item["question"],
                                "model": model_spec,
                                "reference": item["reference"],