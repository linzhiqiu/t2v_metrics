# input_file = "genai_image_results/clip-flant5-xxl_all_tags_528.json"
# input_file = "genai_video_results/clip-flant5-xxl_all_tags_528.json"

# input_file = "genai_image_results/openai:ViT-L-14-336_all_tags_528.json"
input_file = "genai_video_results/openai:ViT-L-14-336_all_tags_528.json"

import json
def percentage_to_decimal(percentage_string):
    """
    Convert a percentage string to a decimal representation.
    
    Args:
    - percentage_string (str): The percentage string to convert (e.g., '64.9%').
    
    Returns:
    - float: The decimal representation of the percentage (e.g., 0.649).
    """
    # Remove '%' and convert the remaining string to float
    number_float = float(percentage_string.replace('%', ''))
    # Convert to decimal by dividing by 100
    decimal_representation = number_float / 100
    return decimal_representation


# Load the data
data = json.load(open(input_file))
tag_groups = {
    'basic': ['basic'],
    'advanced': ['advanced'],
    'basic_all': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation'],
    'advanced_all': ['counting', 'comparison', 'differentiation', 'negation', 'universal'],
}

scores = ['mean', 'human_mean']

# models = ['SDXL_2_1', 'SDXL_Base', 'SDXL_Turbo', 'DeepFloyd_I_XL_v1', 'Midjourney_6', 'DALLE_3']

# from dataset import GenAIBench_Image

# dataset = GenAIBench_Image(root_dir="datasets")

models = ['Modelscope', 'Floor33', 'Pika_v1', 'Gen2', 'Zeroscope']

from dataset import GenAIBench_Video

dataset = GenAIBench_Video(root_dir="datasets", filename='genai_video_final')

for score in scores:
    for _, tags in tag_groups.items():
        for model in models:
            print(f"{model} ({score}):")
            tag_scores = []
            
            for tag in tags:
                if score == 'mean':
                # tag_scores.append(str(data["finegrained"][tag][model][score].replace('%', '')))
                    # score_str = "0."+str(data["genai_skills"][tag][model][score].replace('%', '').replace('.', ''))
                    score_str = f"{percentage_to_decimal(data['genai_skills'][tag][model][score]):.2f}"
                else:
                    score_str = data["genai_skills"][tag][model][score]
                tag_scores.append(score_str)
            
            tag_scores_str = " & ".join(tag_scores)
            print(f"{tags}: {tag_scores_str}")