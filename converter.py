import json
import os
from pathlib import Path

def convert_to_sharegpt4v_custom(input_files, output_file, question_transformer=None):
    """
    Convert JSON files to ShareGPT-4V format with custom question transformation
    
    Args:
        input_files: List of input JSON file paths
        output_file: Output JSON file path
        question_transformer: Function to transform questions (optional)
    """
    
    def default_transformer(question):
        """Default: just add <video> prefix"""
        return f"<video>{question}"
    
    def simple_camera_transformer(question):
        """Simplified camera movement question"""
        return "<video>Is the camera stationary with minor vibrations or shaking? Please only answer Yes or No."
    
    # Use custom transformer or default
    transform_question = question_transformer or default_transformer
    
    sharegpt_data = []
    
    # Process each input file
    for file_path in input_files:
        print(f"Processing {file_path}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert each item in the file
        for item in data:
            # Transform the question
            transformed_question = transform_question(item['Question'])
            
            # Create ShareGPT-4V format entry
            sharegpt_entry = {
                "messages": [
                    {
                        "content": transformed_question,
                        "role": "user"
                    },
                    {
                        "content": item['Answer'],
                        "role": "assistant"
                    }
                ],
                "videos": [
                    item['Video']
                ]
            }
            
            sharegpt_data.append(sharegpt_entry)
    
    # Write output file
    with open(output_file, 'w') as f:
        json.dump(sharegpt_data, f, indent=2)
    
    print(f"✅ Converted {len(sharegpt_data)} entries to {output_file}")
    return len(sharegpt_data)

def main():
    # Configuration
    input_files = [
        "cambench_vqa.json",  # Replace with your actual file paths
        "cambench_vqa_complex.json"
    ]
    output_file = "sharegpt4v_camera_movement.json"
    
    # Custom question transformer (optional)
    def camera_movement_transformer(original_question):
        """Convert long camera movement questions to short format"""
        if "camera movement" in original_question.lower():
            return "<video>Is the camera stationary with minor vibrations or shaking? Please only answer Yes or No."
        else:
            return f"<video>{original_question}"
    
    # Check if input files exist
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Error: Files not found: {missing_files}")
        print("Please update the file paths in the script.")
        return
    
    # Convert files
    print("🔄 Converting to ShareGPT-4V format...")
    total_entries = convert_to_sharegpt4v_custom(
        input_files, 
        output_file, 
        question_transformer=camera_movement_transformer
    )
    
    print(f"\n📊 Summary:")
    print(f"   Input files: {len(input_files)}")
    print(f"   Total entries converted: {total_entries}")
    print(f"   Output file: {output_file}")
    
    # Show a sample of the output
    print(f"\n📝 Sample output:")
    with open(output_file, 'r') as f:
        sample_data = json.load(f)
        if sample_data:
            print(json.dumps(sample_data[0], indent=2))

if __name__ == "__main__":
    main()