# Evaluate on all dataset using a specific score
# CUDA_VISIBLE_DEVICES=1 python genai_image_eval.py --model clip-flant5-xxl
# CUDA_VISIBLE_DEVICES=2 python genai_image_eval.py --model llava-v1.5-13b
# CUDA_VISIBLE_DEVICES=3 python genai_image_eval.py --model instructblip-flant5-xxl
# CUDA_VISIBLE_DEVICES=4 python genai_image_eval.py --model image-reward-v1
# CUDA_VISIBLE_DEVICES=5 python genai_image_eval.py --model pickscore-v1
# CUDA_VISIBLE_DEVICES=5 python genai_image_eval.py --model hpsv2
# CUDA_VISIBLE_DEVICES=5 python genai_image_eval.py --model blip2-itm
# CUDA_VISIBLE_DEVICES=5 python genai_image_eval.py --model openai:ViT-L-14-336
# (skip) CUDA_VISIBLE_DEVICES=5 python genai_image_eval.py --model openai:ViT-B-32
import argparse
import os
import t2i_metrics
from dataset import GenAIBench_Image
import json
import torch

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2i_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model", default="clip-flant5-xxl", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    return parser.parse_args()


def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    result_dir = "genai_image_results"
    os.makedirs(result_dir, exist_ok=True)
    tag_result = {}
    result_path = f"{result_dir}/{args.model}_all_528.pt"
    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists. Skipping.")
        scores = torch.load(result_path)
        our_scores = scores.mean(axis=1)
        import numpy as np
        # for tag_file_name in ['finegrained']:
        for tag_file_name in ['genai_skills']:
            tag_result[tag_file_name] = {}
            tag_file = f"datasets/GenAI-Image-All/{tag_file_name}.json"
            tags = json.load(open(tag_file))
            dataset = GenAIBench_Image(root_dir=args.root_dir)
            human_scores = [np.array(dataset.images[idx]['human_alignment']).mean() for idx in range(len(dataset.images))]
            images_by_model_tag = {}
            for tag in tags:
                images_by_model_tag[tag] = {}
                for prompt_idx in tags[tag]:
                    for image_idx in dataset.prompt_to_images[f"{prompt_idx:05d}"]:
                        model = dataset.images[image_idx]['model']
                        if model not in images_by_model_tag[tag]:
                            images_by_model_tag[tag][model] = []
                        images_by_model_tag[tag][model].append(image_idx)
            
            for tag in tags:
                print(f"Tag: {tag}")
                tag_result[tag_file_name][tag] = {}
                for model in images_by_model_tag[tag]:
                    our_scores_mean = our_scores[images_by_model_tag[tag][model]].mean()
                    our_scores_std = our_scores[images_by_model_tag[tag][model]].std()
                    print(f"{model} (Metric Score): {our_scores_mean:.1%} +- {our_scores_std:.1%}")
                    human_scores_mean = np.array(human_scores)[images_by_model_tag[tag][model]].mean()
                    human_scores_std = np.array(human_scores)[images_by_model_tag[tag][model]].std()
                    print(f"{model} (Human Score): {human_scores_mean:.1%} +- {human_scores_std:.1%}")
                    tag_result[tag_file_name][tag][model] = {'mean': f"{our_scores_mean:.1%}", 'std': f"{our_scores_std:.1%}", 'human_mean': f"{human_scores_mean:.1f}", 'human_std': f"{human_scores_std:.1f}"}
                print()
        
        print("All")
        tag_result['all'] = {}
        all_models = images_by_model_tag[tag]
        for model in all_models:
            all_model_indices = set()
            for tag in images_by_model_tag:
                all_model_indices = all_model_indices.union(set(images_by_model_tag[tag][model]))
            all_model_indices = list(all_model_indices)
            our_scores_mean = our_scores[all_model_indices].mean()
            our_scores_std = our_scores[all_model_indices].std()
            print(f"{model} (Metric Score): {our_scores_mean:.1%} +- {our_scores_std:.1%}")
            human_scores_mean = np.array(human_scores)[all_model_indices].mean()
            human_scores_std = np.array(human_scores)[all_model_indices].std()
            print(f"{model} (Human Score): {human_scores_mean:.1%} +- {human_scores_std:.1%}")
            tag_result['all'][model] = {'mean': f"{our_scores_mean:.1%}", 'std': f"{our_scores_std:.1%}", 'human_mean': f"{human_scores_mean:.1f}", 'human_std': f"{human_scores_std:.1f}"}
        json.dump(tag_result, open(f"genai_image_results/{args.model}_all_tags_528.json", "w"), indent=4)
        return
    
    score_func = t2i_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

    kwargs = {}
    if args.question is not None:
        print(f"Using question template: {args.question}")
        kwargs['question_template'] = args.question
    if args.answer is not None:
        print(f"Using answer template: {args.answer}")
        kwargs['answer_template'] = args.answer
    
    
    print(f"Performance of {args.model}.")
    for dataset_cls in [
        GenAIBench_Image
    ]:
        dataset = dataset_cls(root_dir=args.root_dir)
        scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
        torch.save(scores, result_path)
        results = dataset.evaluate_scores(scores)

if __name__ == "__main__":
    main()
