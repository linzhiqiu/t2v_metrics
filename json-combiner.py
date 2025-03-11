import json
import os
import argparse
from typing import List, Dict, Any
import pandas as pd


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def get_model_name_from_file(file_path: str, data: List[Dict[str, Any]]) -> str:
    """
    Extract the model name from the data or fallback to filename-based approach.
    
    Args:
        file_path: Path to the JSON file
        data: The loaded JSON data
        
    Returns:
        Model name
    """
    # First try to extract model name from the data
    if data and len(data) > 0:
        # Get all keys from the first item
        keys = set(data[0].keys())
        # Find keys that are not common fields
        common_fields = {'video', 'question', 'reference'}
        model_keys = keys - common_fields
        
        # If there's exactly one model key, use it
        if len(model_keys) == 1:
            return list(model_keys)[0]
    
    # Fallback to filename
    return os.path.basename(file_path).split('_')[0]


def combine_json_files(file_paths: List[str], output_file: str, model_mapping: Dict[str, str] = None) -> None:
    """
    Combine multiple JSON files into a single JSON file.
    
    Args:
        file_paths: List of paths to JSON files
        output_file: Path to the output JSON file
        model_mapping: Dictionary mapping file names to model names (optional)
    """
    combined_data = []
    
    # Process each file
    for file_path in file_paths:
        data = load_json_file(file_path)
        if not data:
            continue
        
        # Get model name
        if model_mapping and os.path.basename(file_path) in model_mapping:
            model_name = model_mapping[os.path.basename(file_path)]
        else:
            model_name = get_model_name_from_file(file_path, data)
        
        print(f"Processing file: {file_path} (Model: {model_name})")
        
        # Find the actual model field name in the data
        model_field = None
        if data and len(data) > 0:
            common_fields = {'video', 'question', 'reference'}
            all_fields = set(data[0].keys())
            potential_model_fields = all_fields - common_fields
            
            if len(potential_model_fields) == 1:
                model_field = list(potential_model_fields)[0]
            else:
                print(f"Warning: Could not automatically determine model field in {file_path}. Fields found: {potential_model_fields}")
                continue
        
        # Process each item in the file
        for item in data:
            # Check if there's already an entry for this video
            existing_entry = None
            for entry in combined_data:
                if entry['video'] == item['video']:
                    existing_entry = entry
                    break
            
            if existing_entry:
                # Add this model's caption to the existing entry
                if model_field and model_field in item:
                    existing_entry[model_name] = item[model_field]
            else:
                # Create a new entry
                new_entry = {
                    'video': item['video'],
                    'question': item['question'],
                    'reference': item['reference']
                }
                
                # Add the model's output
                if model_field and model_field in item:
                    new_entry[model_name] = item[model_field]
                
                combined_data.append(new_entry)
    
    # Save combined data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined {len(file_paths)} files into {output_file} with {len(combined_data)} entries")


def generate_summary(output_file: str) -> None:
    """
    Generate a summary of the combined JSON file.
    
    Args:
        output_file: Path to the combined JSON file
    """
    data = load_json_file(output_file)
    if not data:
        return
    
    # Count videos
    num_videos = len(data)
    
    # Count models
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    model_keys = all_keys - {'video', 'question', 'reference'}
    num_models = len(model_keys)
    
    # Count entries per model
    model_counts = {model: 0 for model in model_keys}
    for item in data:
        for model in model_keys:
            if model in item:
                model_counts[model] += 1
    
    print("\nSummary:")
    print(f"Total videos: {num_videos}")
    print(f"Total models: {num_models}")
    print("Entries per model:")
    for model, count in model_counts.items():
        print(f"  {model}: {count}")


def export_to_excel(output_file: str, excel_file: str) -> None:
    """
    Export the combined JSON file to Excel for easier viewing.
    
    Args:
        output_file: Path to the combined JSON file
        excel_file: Path to the output Excel file
    """
    data = load_json_file(output_file)
    if not data:
        return
    
    df = pd.DataFrame(data)
    df.to_excel(excel_file, index=False)
    print(f"Exported data to Excel file: {excel_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple JSON files of caption evaluations")
    parser.add_argument("--files", nargs="+", required=True, help="List of JSON files to combine")
    parser.add_argument("--output", default="combined_results.json", help="Output JSON file")
    parser.add_argument("--excel", default=None, help="Also export to Excel file (requires pandas)")
    parser.add_argument("--mapping", nargs="+", help="Model mapping in format filename:modelname")
    
    args = parser.parse_args()
    
    # Process model mapping if provided
    model_mapping = {}
    if args.mapping:
        for mapping in args.mapping:
            if ":" in mapping:
                filename, modelname = mapping.split(":", 1)
                model_mapping[filename] = modelname
    
    # Combine files
    combine_json_files(args.files, args.output, model_mapping)
    
    # Generate summary
    generate_summary(args.output)
    
    # Export to Excel if requested
    if args.excel:
        try:
            export_to_excel(args.output, args.excel)
        except ImportError:
            print("Error: pandas is required for Excel export. Install with 'pip install pandas openpyxl'.")