# import json
# import os
# import argparse
# import numpy as np
# from collections import Counter
# import string
# from tqdm import tqdm
# import pandas as pd
# import openai
# import time
# import re
# from typing import List, Dict, Any, Tuple, Set


# def load_json_file(file_path: str) -> List[Dict[str, Any]]:
#     """Load data from a JSON file."""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         return data
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return []


# def preprocess_text(text):
#     """Preprocess text for evaluation"""
#     # Handle None or empty string
#     if not text:
#         return []
    
#     # Convert to lowercase
#     text = text.lower()
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Split into words
#     words = text.split()
#     return words


# def calculate_spice_score(reference, candidate):
#     """Simplified SPICE calculation"""
#     # Handle None or empty values
#     if not reference or not candidate:
#         return 0.0
    
#     # Preprocess texts
#     ref_words = set(preprocess_text(reference))
#     cand_words = set(preprocess_text(candidate))
    
#     # Calculate precision and recall
#     if len(cand_words) == 0:
#         return 0.0
    
#     intersection = ref_words.intersection(cand_words)
#     precision = len(intersection) / len(cand_words)
#     recall = len(intersection) / len(ref_words) if len(ref_words) > 0 else 0
    
#     # Calculate F1 score
#     if precision + recall == 0:
#         return 0.0
#     f1 = 2 * precision * recall / (precision + recall)
    
#     return f1


# def calculate_cider_score(reference, candidate):
#     """Simplified CIDEr calculation"""
#     # Handle None or empty values
#     if not reference or not candidate:
#         return 0.0
    
#     # Preprocess texts
#     ref_words = preprocess_text(reference)
#     cand_words = preprocess_text(candidate)
    
#     # Calculate word counts
#     ref_counts = Counter(ref_words)
#     cand_counts = Counter(cand_words)
    
#     # Calculate cosine similarity
#     all_words = set(ref_counts.keys()).union(set(cand_counts.keys()))
    
#     if not all_words:
#         return 0.0
    
#     dot_product = sum(ref_counts[word] * cand_counts[word] for word in all_words)
#     ref_magnitude = np.sqrt(sum(ref_counts[word] ** 2 for word in ref_counts))
#     cand_magnitude = np.sqrt(sum(cand_counts[word] ** 2 for word in cand_counts))
    
#     if ref_magnitude == 0 or cand_magnitude == 0:
#         return 0.0
    
#     similarity = dot_product / (ref_magnitude * cand_magnitude)
    
#     return similarity


# def calculate_generative_match(reference, candidate, api_key=None, retries=3, delay=2):
#     """
#     Calculate generative match score using GPT-4o as judge.
    
#     Args:
#         reference: Reference caption
#         candidate: Model-generated caption
#         api_key: OpenAI API key
#         retries: Number of retries if API call fails
#         delay: Delay between retries in seconds
        
#     Returns:
#         Logit probability for "Yes" response
#     """
#     if not reference or not candidate:
#         return 0.0
    
#     # Set up OpenAI API
#     if api_key:
#         openai.api_key = api_key
#     else:
#         openai.api_key = os.environ.get("OPENAI_API_KEY")
    
#     if not openai.api_key:
#         print("Warning: No OpenAI API key provided. Returning placeholder score.")
#         return 0.5  # Return placeholder score
    
#     prompt = f"Reference caption: '{reference}'\nCandidate caption: '{candidate}'\n\nDoes the candidate caption match the reference caption? Answer Yes or No."
    
#     for attempt in range(retries):
#         try:
#             # Call GPT-4o API
#             response = openai.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0,
#                 max_tokens=5,
#                 logprobs=True,
#                 top_logprobs=5
#             )
            
#             # Extract response and logprobs
#             content = response.choices[0].message.content.strip().lower()
            
#             # Check if the answer is clearly yes or no
#             if content.startswith("yes"):
#                 return 1.0
#             elif content.startswith("no"):
#                 return 0.0
            
#             # If we can't determine from the content, try to get the logprobs
#             logprobs = response.choices[0].logprobs.content[0].top_logprobs
            
#             # Look for "yes" in top logprobs
#             for token_info in logprobs:
#                 if token_info.token.strip().lower() == "yes":
#                     return np.exp(token_info.logprob)  # Convert log probability to probability
            
#             # If "yes" not found in top logprobs, return low probability
#             return 0.1
            
#         except Exception as e:
#             print(f"Error calling OpenAI API (attempt {attempt+1}/{retries}): {str(e)}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries exceeded. Using fallback score.")
#                 return 0.5  # Fallback score
    
#     return 0.5  # Should never reach here but just in case


# def evaluate_models_from_combined(combined_data, api_key=None):
#     """
#     Evaluate models from a combined JSON file.
    
#     Args:
#         combined_data: List of dictionaries with combined model outputs
#         api_key: OpenAI API key for GPT-4o judge
        
#     Returns:
#         Dictionary mapping model names to their evaluation results
#     """
#     # Extract model names (fields that are not common fields)
#     common_fields = {"video", "question", "reference"}
    
#     if len(combined_data) == 0:
#         print("Error: No data found in the input")
#         return {}
    
#     all_fields = set(combined_data[0].keys())
#     model_names = list(all_fields - common_fields)
    
#     if not model_names:
#         print("Error: No model outputs found in the data")
#         return {}
    
#     print(f"Found {len(model_names)} models: {', '.join(model_names)}")
    
#     # Prepare data structure for results
#     results = {}
    
#     # For each model, calculate metrics
#     for model_name in model_names:
#         print(f"\nEvaluating model: {model_name}")
        
#         spice_scores = []
#         cider_scores = []
#         gen_match_scores = []
        
#         # Process each item
#         for item in tqdm(combined_data, desc=f"Processing {model_name}"):
#             reference = item.get("reference", "")
            
#             # Skip if this model doesn't have a caption for this item
#             if model_name not in item:
#                 continue
                
#             candidate = item.get(model_name, "")
            
#             # Skip items with missing data
#             if not reference or not candidate:
#                 continue
                
#             # Calculate SPICE score
#             spice = calculate_spice_score(reference, candidate)
#             spice_scores.append(spice)
            
#             # Calculate CIDEr score
#             cider = calculate_cider_score(reference, candidate)
#             cider_scores.append(cider)
            
#             # Calculate generative match score
#             if api_key:
#                 gen_match = calculate_generative_match(reference, candidate, api_key)
#                 gen_match_scores.append(gen_match)
        
#         # Calculate averages
#         if spice_scores:
#             avg_spice = float(np.mean(spice_scores))
#         else:
#             avg_spice = 0.0
            
#         if cider_scores:
#             avg_cider = float(np.mean(cider_scores))
#         else:
#             avg_cider = 0.0
            
#         if gen_match_scores:
#             avg_gen_match = float(np.mean(gen_match_scores))
#         else:
#             avg_gen_match = 0.0
        
#         # Store results
#         results[model_name] = {
#             "model": model_name,
#             "spice": avg_spice,
#             "cider": avg_cider,
#             "gen_match": avg_gen_match if gen_match_scores else None,
#             "count": len(spice_scores)
#         }
    
#     return results


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate model outputs in a combined JSON file")
#     parser.add_argument("--input", type=str, required=True, help="Path to combined JSON file")
#     parser.add_argument("--output", default="evaluation_results.json", help="Path to output JSON file")
#     parser.add_argument("--excel", default="evaluation_results.xlsx", help="Path to output Excel file")
#     parser.add_argument("--api_key", type=str, help="OpenAI API key for GPT-4o judge")
#     parser.add_argument("--no_gpt", action="store_true", help="Skip GPT-4o judge evaluation")
    
#     args = parser.parse_args()
    
#     # Load the combined data
#     combined_data = load_json_file(args.input)
#     if not combined_data:
#         print("Error: Could not load data from input file")
#         return
    
#     # Evaluate models
#     api_key = args.api_key if not args.no_gpt else None
#     evaluation_results = evaluate_models_from_combined(combined_data, api_key)
    
#     # Convert results to list for easier serialization
#     results_list = list(evaluation_results.values())
    
#     # Print results
#     print("\nEvaluation Results:")
#     for model_name, result in evaluation_results.items():
#         print(f"Model: {model_name}")
#         print(f"  SPICE: {result['spice']:.4f}")
#         print(f"  CIDEr: {result['cider']:.4f}")
#         if result['gen_match'] is not None:
#             print(f"  GPT-4o Judge Score: {result['gen_match']:.4f}")
#         print(f"  Sample count: {result['count']}")
    
#     # Save results to JSON
#     with open(args.output, "w") as f:
#         json.dump(results_list, f, indent=2)
#     print(f"\nSaved evaluation results to {args.output}")
    
#     # Export to Excel
#     try:
#         # Create DataFrame for evaluation results
#         results_df = pd.DataFrame(results_list)
        
#         # Create a model comparison DataFrame sorted by generative match score
#         if any(r['gen_match'] is not None for r in results_list):
#             sort_key = 'gen_match'
#         else:
#             sort_key = 'cider'
        
#         comparison_df = results_df.sort_values(by=sort_key, ascending=False)
        
#         # Export to Excel with multiple sheets
#         with pd.ExcelWriter(args.excel) as writer:
#             results_df.to_excel(writer, sheet_name='All Results', index=False)
#             comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
            
#             # Add a sheet with detailed item-level results for further analysis
#             if len(combined_data) <= 1000:  # Only if the data is not too large
#                 detailed_results = []
                
#                 for item in combined_data:
#                     video = item.get("video", "")
#                     question = item.get("question", "")
#                     reference = item.get("reference", "")
                    
#                     for model_name in evaluation_results.keys():
#                         if model_name in item:
#                             candidate = item.get(model_name, "")
                            
#                             spice = calculate_spice_score(reference, candidate)
#                             cider = calculate_cider_score(reference, candidate)
                            
#                             detailed_results.append({
#                                 "video": video,
#                                 "question": question,
#                                 "model": model_name,
#                                 "reference": reference,
#                                 "candidate": candidate,
#                                 "spice": spice,
#                                 "cider": cider
#                             })
                
#                 # Create DataFrame and sort by model and video
#                 detailed_df = pd.DataFrame(detailed_results)
#                 detailed_df = detailed_df.sort_values(by=["model", "video"])
                
#                 # Export to the Excel file
#                 detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
#         print(f"Exported results to Excel file: {args.excel}")
#     except ImportError:
#         print("Warning: pandas is required for Excel export. Install with 'pip install pandas openpyxl'.")


# if __name__ == "__main__":
#     main()

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


def calculate_generative_match(reference, candidate, api_key='api_key', retries=3, delay=2):
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
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai.api_key:
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
    common_fields = {"video", "question", "reference"}
    
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
            reference = item.get("reference", "")
            
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs in a combined JSON file")
    parser.add_argument("--input", type=str, required=True, help="Path to combined JSON file")
    parser.add_argument("--output", default="evaluation_results_72b.json", help="Path to output JSON file")
    parser.add_argument("--excel", default="evaluation_results_72b.xlsx", help="Path to output Excel file")
    parser.add_argument("--api_key", type=str, help="OpenAI API key for GPT-4o judge")
    parser.add_argument("--no_gpt", action="store_true", help="Skip GPT-4o judge evaluation")
    
    args = parser.parse_args()
    
    # Check for required NLTK data
    try:
        nltk.data.find('punkt')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
    
    # Load the combined data
    combined_data = load_json_file(args.input)
    if not combined_data:
        print("Error: Could not load data from input file")
        return
    
    # Evaluate models
    api_key = args.api_key if not args.no_gpt else None
    evaluation_results = evaluate_models_from_combined(combined_data, api_key)
    
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
    with open(args.output, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\nSaved evaluation results to {args.output}")
    
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
        with pd.ExcelWriter(args.excel) as writer:
            results_df.to_excel(writer, sheet_name='All Results', index=False)
            comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
            
            # Add a sheet with detailed item-level results for further analysis
            if len(combined_data) <= 1000:  # Only if the data is not too large
                detailed_results = []
                
                for item in combined_data:
                    video = item.get("video", "")
                    question = item.get("question", "")
                    reference = item.get("reference", "")
                    
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
        
        print(f"Exported results to Excel file: {args.excel}")
    except ImportError:
        print("Warning: pandas is required for Excel export. Install with 'pip install pandas openpyxl'.")


if __name__ == "__main__":
    main()