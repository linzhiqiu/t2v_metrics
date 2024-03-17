import os
import json
import csv
import pandas as pd

def read_csv_file(file_path):
    """Read a .csv file and return a list of lines."""
    csv_file = pd.read_csv(file_path)
    return csv_file.to_dict(orient='records')

# Models and prompts setup
models = ['floor33', 'gen2', 'pika', 'modelscope', 'zeroscope']
prompts = read_csv_file("datasets/t2vscore/labels.csv")

scores = {
    'text_alignment_score': 'datasets/t2vscore_alignment_score.json',
    'video_quality_score': 'datasets/t2vscore_quality_score.json',
}

for score_key, score_file in scores.items():
    d = {}
    # Generate Markdown files
    for item in prompts:
        video_id = item['video_id']
        model_name = item['model_name']
        prompt = item['prompt']
        score = float(item[score_key])
        if video_id not in d:
            d[video_id] = {
                'prompt': prompt,
                'models': {},
            }
        
        assert d[video_id]['prompt'] == prompt
        d[video_id]['models'][model_name] = [score]

    json.dump(d, open(score_file, 'w'), indent=4)