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
    clip_flant5_result_path = f"{result_dir}/clip-flant5-xxl_all_528.pt"
    clip_result_path = f"{result_dir}/openai:ViT-L-14-336_all_528.pt"
    assert os.path.exists(clip_flant5_result_path) and os.path.exists(clip_result_path)
    vqa_scores = torch.load(clip_flant5_result_path)
    clip_scores = torch.load(clip_result_path)
    vqa_scores = vqa_scores.mean(axis=1)
    clip_scores = clip_scores.mean(axis=1)
    import numpy as np
    dataset = GenAIBench_Image(root_dir=args.root_dir)
    human_scores = [np.array(dataset.images[idx]['human_alignment']).mean() for idx in range(len(dataset.images))]
    images = {}
    for prompt_idx in dataset.dataset:
        for image_idx in dataset.prompt_to_images[prompt_idx]:
            model = dataset.images[image_idx]['model']
            if model not in images:
                images[model] = {}
            images[model][prompt_idx] = image_idx
    
    new_dataset = {}
    keys_to_delete = ['SDXL_Turbo', 'SDXL_2_1']
    for key in keys_to_delete:
        del images[key]
    for prompt_idx in dataset.dataset:
        new_dataset[prompt_idx] = dataset.dataset[prompt_idx]
        for key in keys_to_delete:
            del new_dataset[prompt_idx]['models'][key]
        clip_flant5_scores = [float(vqa_scores[images[model][prompt_idx]]) for model in images]
        clipscores = [float(clip_scores[images[model][prompt_idx]]) for model in images]
        humanscores = [human_scores[images[model][prompt_idx]] for model in images]
        clip_flant5_rankings = np.argsort(clip_flant5_scores)[::-1]
        clip_rankings = np.argsort(clipscores)[::-1]
        human_rankings = np.argsort(humanscores)[::-1]
        
        # only save if clip_flant5_rankings agrees with human
        if clip_flant5_rankings.tolist() != human_rankings.tolist():
            del new_dataset[prompt_idx]
            continue
        
        for model_idx, model in enumerate(images):
            new_dataset[prompt_idx]['models'][model] = {
                'clip_flant5': f"{float(vqa_scores[images[model][prompt_idx]]):.3f}",
                'clip': f"{float(clip_scores[images[model][prompt_idx]]):.3f}",
                'human': f"{human_scores[images[model][prompt_idx]]:.2f}",
                'clip_flant5_rank': f"{clip_flant5_rankings.tolist().index(model_idx) + 1}",
                'clip_rank': f"{clip_rankings.tolist().index(model_idx) + 1}",
                'human_rank': f"{human_rankings.tolist().index(model_idx) + 1}",
                # 'human_all': dataset.dataset[prompt_idx]['models'][model],
            }
    with open("genai_image_teaser.json", "w") as f:
        json.dump(new_dataset, f, indent=4)
    # for tag in tags:
    #     print(f"Tag: {tag}")
    #     tag_result[tag_file_name][tag] = {}
    #     for model in images_by_model_tag[tag]:
    #         our_scores_mean = our_scores[images_by_model_tag[tag][model]].mean()
    #         our_scores_std = our_scores[images_by_model_tag[tag][model]].std()
    #         print(f"{model} (Metric Score): {our_scores_mean:.1%} +- {our_scores_std:.1%}")
    #         human_scores_mean = np.array(human_scores)[images_by_model_tag[tag][model]].mean()
    #         human_scores_std = np.array(human_scores)[images_by_model_tag[tag][model]].std()
    #         print(f"{model} (Human Score): {human_scores_mean:.1%} +- {human_scores_std:.1%}")
    #         tag_result[tag_file_name][tag][model] = {'mean': f"{our_scores_mean:.1%}", 'std': f"{our_scores_std:.1%}", 'human_mean': f"{human_scores_mean:.1f}", 'human_std': f"{human_scores_std:.1f}"}
    #     print()
        
    #     print("All")
    #     tag_result['all'] = {}
    #     all_models = images_by_model_tag[tag]
    #     for model in all_models:
    #         all_model_indices = set()
    #         for tag in images_by_model_tag:
    #             all_model_indices = all_model_indices.union(set(images_by_model_tag[tag][model]))
    #         all_model_indices = list(all_model_indices)
    #         our_scores_mean = our_scores[all_model_indices].mean()
    #         our_scores_std = our_scores[all_model_indices].std()
    #         print(f"{model} (Metric Score): {our_scores_mean:.1%} +- {our_scores_std:.1%}")
    #         human_scores_mean = np.array(human_scores)[all_model_indices].mean()
    #         human_scores_std = np.array(human_scores)[all_model_indices].std()
    #         print(f"{model} (Human Score): {human_scores_mean:.1%} +- {human_scores_std:.1%}")
    #         tag_result['all'][model] = {'mean': f"{our_scores_mean:.1%}", 'std': f"{our_scores_std:.1%}", 'human_mean': f"{human_scores_mean:.1f}", 'human_std': f"{human_scores_std:.1f}"}
    #     json.dump(tag_result, open(f"genai_image_results/{args.model}_all_tags_528.json", "w"), indent=4)
    #     return
    
    # score_func = t2i_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

    # kwargs = {}
    # if args.question is not None:
    #     print(f"Using question template: {args.question}")
    #     kwargs['question_template'] = args.question
    # if args.answer is not None:
    #     print(f"Using answer template: {args.answer}")
    #     kwargs['answer_template'] = args.answer
    
    
    # print(f"Performance of {args.model}.")
    # for dataset_cls in [
    #     GenAIBench_Image
    # ]:
    #     dataset = dataset_cls(root_dir=args.root_dir)
    #     scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
    #     torch.save(scores, result_path)
    #     results = dataset.evaluate_scores(scores)

if __name__ == "__main__":
    main()
