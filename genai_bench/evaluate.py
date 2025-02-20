# Evaluate your own model on GenAI-Bench-Image (with 1600 prompt)
# Example scripts to run:
# VQAScore: python genai_bench/evaluate.py --model clip-flant5-xxl --output_dir ./outputs --gen_model runwayml/stable-diffusion-v1-5
# GPT4o VQAScore: python genai_bench/evaluate.py --model gpt-4o --output_dir ./outputs --gen_model runwayml/stable-diffusion-v1-5
import argparse
import os
import t2v_metrics
from dataset import GenAIBench_Image
import json
import torch

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--output_dir", default="./outputs", type=str,
                        help="Directory where you have saved your model's output.")
    parser.add_argument("--gen_model", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_prompts", default=1600, type=int, choices=[527, 1600])
    parser.add_argument("--model", default="clip-flant5-xxl", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--result_dir", default="./genai_bench_results", type=str)
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

def show_performance_per_skill(our_scores, dataset, gen_model='runwayml/stable-diffusion-v1-5', print_std=False, tag_groups=tag_groups):
    tag_result = {}
    tag_file = f"{dataset.root_dir}/genai_skills.json"
    tags = json.load(open(tag_file))
    prompt_to_items = {prompt_idx: [int(prompt_idx)] for prompt_idx in dataset.dataset.keys()}
    items_by_model_tag = {}
    for tag in tags:
        items_by_model_tag[tag] = {}
        for prompt_idx in tags[tag]:
            for image_idx in prompt_to_items[f"{prompt_idx:05d}"]:
                if gen_model not in items_by_model_tag[tag]:
                    items_by_model_tag[tag][gen_model] = []
                items_by_model_tag[tag][gen_model].append(image_idx)
    
    for tag in tags:
        # print(f"Tag: {tag}")
        tag_result[tag] = {}
        our_scores_mean = our_scores[items_by_model_tag[tag][gen_model]].mean()
        our_scores_std = our_scores[items_by_model_tag[tag][gen_model]].std()
        tag_result[tag][gen_model] = {
            'metric': {'mean': our_scores_mean, 'std': our_scores_std},
        }
        # print()
        
    tag_result['all'] = {}
    all_model_indices = set()
    for tag in items_by_model_tag:
        all_model_indices = all_model_indices.union(set(items_by_model_tag[tag][gen_model]))
    all_model_indices = list(all_model_indices)
    our_scores_mean = our_scores[all_model_indices].mean()
    our_scores_std = our_scores[all_model_indices].std()
    tag_result['all'][gen_model] = {
        'metric': {'mean': our_scores_mean, 'std': our_scores_std},
    }
    
    for tag_group in tag_groups:
        print(f"Tag Group: {tag_group} ({'metric'} performance)")
        tag_header = f"{'Model':<20}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
        print(tag_header)
        if print_std:
            detailed_scores = [f"{tag_result[tag][gen_model]['metric']['mean']:.2f} +- {tag_result[tag][gen_model]['metric']['std']:.2f}" for tag in tag_groups[tag_group]]
        else:
            detailed_scores = [f"{tag_result[tag][gen_model]['metric']['mean']:.2f}" for tag in tag_groups[tag_group]]
        detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
        model_scores = f"{gen_model:<20}" + detailed_scores
        print(model_scores)



def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    result_dir = os.path.join(args.result_dir, args.gen_model)
    os.makedirs(result_dir, exist_ok=True)
    
    if not os.path.exists(os.path.join(args.output_dir, args.gen_model)):
        raise ValueError(f"Output directory {os.path.join(args.output_dir, args.gen_model)} does not exist. Please run python genai_bench/generate.py to generate images.")

    dataset =  GenAIBench_Image(root_dir=args.root_dir, num_prompts=args.num_prompts) # 'num_prompts' is the number of prompts in GenAI-Bench（1600 in GenAI-Bench paper; 527 in VQAScore paper）
    model_output = []
    for prompt_idx in dataset.dataset.keys():
        prompt = dataset.dataset[prompt_idx]['prompt']
        image_path = os.path.join(args.output_dir, args.gen_model, f"{prompt_idx}.jpeg")
        assert os.path.exists(image_path), f"Image {image_path} does not exist. Please run python genai_bench/generate.py to generate images."
        model_output.append({
            'images': [image_path],
            'texts': [prompt]
        })
    
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
    
    result_path = f"{result_dir}/{args.model}_{args.num_prompts}_prompts.pt"
    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists. Skipping.")
        scores = torch.load(result_path)
    else:
        print(f"Scoring {args.model}.")
        scores = score_func.batch_forward(model_output, batch_size=args.batch_size, **kwargs).cpu()
        torch.save(scores, result_path)
        
    
    ### Get performance per skill
    our_scores = scores.mean(axis=1)
    show_performance_per_skill(our_scores, dataset, print_std=True, gen_model=args.gen_model)    
    

if __name__ == "__main__":
    main()
