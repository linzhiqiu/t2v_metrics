# Evaluate on GenAI-Bench-Image (with 527 prompt) using a specific model
# Example scripts to run:
# VQAScore: python genai_image_eval.py --model clip-flant5-xxl
# CLIPScore: python genai_image_eval.py --model openai:ViT-L-14-336
# GPT4o VQAScore: python genai_image_eval.py --model gpt-4o
import argparse
import os
import t2v_metrics
from dataset import GenAIBench_Image
import json
import torch
import numpy as np

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model", default="clip-flant5-xxl", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--result_dir", default="./genai_image_results", type=str)
    parser.add_argument("--openai_key", default=None, type=str)
    parser.add_argument("--openai_key_path", default='./_OPENAI_API_KEY.txt', type=str)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--detail", type=str, default='auto', choices=['low', 'auto', 'high'])
    return parser.parse_args()


tag_groups = {
    'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
    'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced'],
    'overall': ['basic', 'advanced', 'all']
}

def show_performance_per_skill(our_scores, dataset, items_name='images', prompt_to_items_name='prompt_to_images', print_std=False, tag_groups=tag_groups):
    tag_result = {}
    tag_file = f"{dataset.root_dir}/genai_skills.json"
    tags = json.load(open(tag_file))
    items = getattr(dataset, items_name)
    prompt_to_items = getattr(dataset, prompt_to_items_name)
    human_scores = [np.array(items[idx]['human_alignment']).mean() for idx in range(len(items))]
    items_by_model_tag = {}
    for tag in tags:
        items_by_model_tag[tag] = {}
        for prompt_idx in tags[tag]:
            for image_idx in prompt_to_items[f"{prompt_idx:05d}"]:
                model = items[image_idx]['model']
                if model not in items_by_model_tag[tag]:
                    items_by_model_tag[tag][model] = []
                items_by_model_tag[tag][model].append(image_idx)
    
    for tag in tags:
        # print(f"Tag: {tag}")
        tag_result[tag] = {}
        for model in items_by_model_tag[tag]:
            our_scores_mean = our_scores[items_by_model_tag[tag][model]].mean()
            our_scores_std = our_scores[items_by_model_tag[tag][model]].std()
            # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
            human_scores_mean = np.array(human_scores)[items_by_model_tag[tag][model]].mean()
            human_scores_std = np.array(human_scores)[items_by_model_tag[tag][model]].std()
            # print(f"{model} (Human Score): {human_scores_mean:.1f} +- {human_scores_std:.1f}")
            tag_result[tag][model] = {
                'metric': {'mean': our_scores_mean, 'std': our_scores_std},
                'human': {'mean': human_scores_mean, 'std': human_scores_std},
            }
        # print()
        
    # print("All")
    tag_result['all'] = {}
    all_models = items_by_model_tag[tag]
    for model in all_models:
        all_model_indices = set()
        for tag in items_by_model_tag:
            all_model_indices = all_model_indices.union(set(items_by_model_tag[tag][model]))
        all_model_indices = list(all_model_indices)
        our_scores_mean = our_scores[all_model_indices].mean()
        our_scores_std = our_scores[all_model_indices].std()
        # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
        human_scores_mean = np.array(human_scores)[all_model_indices].mean()
        human_scores_std = np.array(human_scores)[all_model_indices].std()
        # print(f"{model} (Human Score): {human_scores_mean:.1f} +- {human_scores_std:.1f}")
        tag_result['all'][model] = {
            'metric': {'mean': our_scores_mean, 'std': our_scores_std},
            'human': {'mean': human_scores_mean, 'std': human_scores_std},
        }
    
    for tag_group in tag_groups:
        for score_name in ['metric', 'human']:
            print(f"Tag Group: {tag_group} ({score_name} performance)")
            tag_header = f"{'Model':<20}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
            print(tag_header)
            for model_name in all_models:
                if print_std:
                    detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f} +- {tag_result[tag][model_name][score_name]['std']:.2f}" for tag in tag_groups[tag_group]]
                else:
                    detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f}" for tag in tag_groups[tag_group]]
                detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
                model_scores = f"{model_name:<20}" + detailed_scores
                print(model_scores)
            print()
        print()



def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    
    os.makedirs(args.result_dir, exist_ok=True)
    dataset = GenAIBench_Image(root_dir=args.root_dir)
    result_path = f"{args.result_dir}/{args.model}_527_prompts.pt"
    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists. Skipping.")
        scores = torch.load(result_path)
    else:
        if args.model in ['gpt-4o', 'gpt-4-turbo']:
            if args.openai_key is None:
                args.openai_key = open(args.openai_key_path, 'r').read().strip()
            assert not (args.openai_key is None and args.openai_key_path is None), "Please provide either openai_key or openai_key_path."
        
            score_func = t2v_metrics.get_score_model(
                model=args.model, device=args.device, cache_dir=args.cache_dir, openai_key=args.openai_key, top_logprobs=args.top_logprobs)
            for item in dataset:
                images = item['images']
                for image in images:
                    assert os.path.getsize(image) < 15 * 1024 * 1024, f"File size of {image} is {os.path.getsize(image)/1048576} bytes, which is larger than 15mb."
                    img_type = image.split('.')[-1]
                    assert img_type in ['png', 'jpeg', 'jpg', 'gif', 'webp'], f"Image type {img_type} is not supported."
        else:
            score_func = t2v_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

        kwargs = {}
        if args.question is not None:
            print(f"Using question template: {args.question}")
            kwargs['question_template'] = args.question
        if args.answer is not None:
            print(f"Using answer template: {args.answer}")
            kwargs['answer_template'] = args.answer
        
        
        print(f"Performance of {args.model}.")
        scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
        torch.save(scores, result_path)
        
    
    
    ### Get performance per skill
    our_scores = scores.mean(axis=1)
    show_performance_per_skill(our_scores, dataset, print_std=True)    
    
    print("Alignment Performance")
    ### Alignment performance
    dataset.evaluate_scores(scores)

if __name__ == "__main__":
    main()
