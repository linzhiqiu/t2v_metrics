# Ranking on GenAI-Bench-Image (with 800 prompt x 9 images) using a specific model
# Example scripts to run:
# python genai_image_ranking.py --model clip-flant5-xxl --gen_model DALLE_3
# python genai_image_ranking.py --model clip-flant5-xxl --gen_model SDXL_Base
# python genai_image_ranking.py --model openai:ViT-L-14-336 --gen_model DALLE_3
# python genai_image_ranking.py --model openai:ViT-L-14-336 --gen_model SDXL_Base
# python genai_image_ranking.py --model pickscore-v1 --gen_model DALLE_3
# python genai_image_ranking.py --model pickscore-v1 --gen_model SDXL_Base
# python genai_image_ranking.py --model image-reward-v1 --gen_model DALLE_3
# python genai_image_ranking.py --model image-reward-v1 --gen_model SDXL_Base
# python genai_image_ranking.py --model clip-flant5-xl --gen_model DALLE_3
# python genai_image_ranking.py --model clip-flant5-xl --gen_model SDXL_Base
# python genai_image_ranking.py --model llava-v1.5-13b --gen_model DALLE_3
# python genai_image_ranking.py --model llava-v1.5-13b --gen_model SDXL_Base

import argparse
import os
import t2v_metrics
from dataset import GenAIBench_Ranking
import json
import torch
import numpy as np

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--model", default="clip-flant5-xxl", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--result_dir", default="./ranking_results", type=str)
    # Ranking specific
    parser.add_argument("--gen_model", default="DALLE_3", type=str, choices=['DALLE_3', 'SDXL_Base'])
    return parser.parse_args()


def compute_scores_per_skill(scores, tags, images_to_prompt_idx):
    prompt_num = scores.shape[0]
    skill_vqascores = {}
    for tag in tags:
        tag_indices = tags[tag]
        tag_score = []
        for idx in range(prompt_num):
            prompt_idx = images_to_prompt_idx[idx]
            if prompt_idx in tag_indices:
                tag_score.append(scores[idx].item())
        tag_score = np.array(tag_score)
        skill_vqascores[tag] = tag_score.mean()

    skill_vqascores['all'] = scores.mean().item() 
    return skill_vqascores

def rerank_human_scores(dataset, rerank_idx_by_9, rerank_idx_by_3, items_name='images'):
    items = getattr(dataset, items_name)

    human_scores_raw = [items[idx]['human_score'] for idx in range(len(items))]
    human_scores_raw = np.array(human_scores_raw)
    human_scores_per_prompt = human_scores_raw.reshape((int(len(items)/9), 9))

    # ------------------- Baseline -------------------
    human_scores_random = human_scores_per_prompt.mean(axis=1)
    # ----------------- Rerank by 9 -----------------
    human_scores_rerank_by9 = human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), rerank_idx_by_9]
    human_oracle_by9 = np.max(human_scores_per_prompt, axis=1)

    # ----------------- Rerank by 3 -----------------
    prompt_num = human_scores_per_prompt.shape[0]
    human_oracle_by3 = []
    human_scores_rerank_by3 = []
    for prompt_idx in range(prompt_num):
        human_score = human_scores_per_prompt[prompt_idx]
        human_tops = []
        rerank_tops = []
        for img_idx in range(0,9,3):
            human_top = np.max(human_score[img_idx:img_idx+3])
            human_tops.append(human_top)
            rerank_top = human_score[rerank_idx_by_3[prompt_idx][img_idx//3]]
            rerank_tops.append(rerank_top)
        
        human_tops = np.array(human_tops)
        human_oracle_by3.append(np.mean(human_tops))
        rerank_tops = np.array(rerank_tops)
        human_scores_rerank_by3.append(np.mean(rerank_tops))
    human_oracle_by3 = np.array(human_oracle_by3)
    human_scores_rerank_by3 = np.array(human_scores_rerank_by3)

    human_scores = {'random': human_scores_random, 'rerank_by_9': human_scores_rerank_by9, 'rerank_by_3': human_scores_rerank_by3, 'human_oracle_by9': human_oracle_by9, 'human_oracle_by3': human_oracle_by3}
    
    return human_scores, human_scores_per_prompt


def compute_top1_acc_skill(vqascores, human_scores, tags, images_to_prompt_idx):
    prompt_num = vqascores.shape[0]
    skill_acc = {}
    skill_set = ["basic", "advanced", "all"]
    for skill in skill_set:
        tag_indices = tags[skill]
        count = 0
        for i in range(prompt_num):
            prompt_idx = images_to_prompt_idx[i]
            if prompt_idx in tag_indices:
                vqa_top_idx = torch.argmax(vqascores[i])
                human_top_idx = np.where(human_scores[i] == np.max(human_scores[i]))
                human_top_idx = list(human_top_idx[0])
                if vqa_top_idx in human_top_idx:
                    count += 1
        acc = count / len(tag_indices)
        skill_acc[skill] = acc
    
    return skill_acc


def compute_top1_acc_group_skill(vqascores, human_scores, tags, images_to_prompt_idx):
    prompt_num = vqascores.shape[0]
    skill_acc_group = {}
    skill_set = ["basic", "advanced", "all"]
    for skill in skill_set:
        tag_indices = tags[skill]
        count = 0
        total = 0
        for i in range(prompt_num):
            prompt_idx = images_to_prompt_idx[i]
            if prompt_idx in tag_indices:
                for img_idx in range(0,9,3):
                    vqa_top_idx = torch.argmax(vqascores[i][img_idx:img_idx+3])
                    human_top_idx = np.where(human_scores[i][img_idx:img_idx+3] == np.max(human_scores[i][img_idx:img_idx+3]))
                    human_top_idx = list(human_top_idx[0])
                    total +=1
                    if vqa_top_idx in human_top_idx:
                        count += 1
        acc = count / total
        skill_acc_group[skill] = acc
    
    return skill_acc_group

def show_performance(args, skill_vqascores, skill_human_scores):
    # assert skill_vqascores[0].keys() == skill_human_scores[0].keys()
    tag_groups = {
    'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
    'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced'],
    'overall': ['basic', 'advanced', 'all']}
    score_names = ['metric', 'human']
    for tag_group in tag_groups:
        for score_name in score_names:
            # print vqascores and human scores in a table
            print(f"Tag Group: {tag_group} ({score_name} performance)")
            tag_header = f"{'Model':<30}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
            print(tag_header)
            if score_name == 'human':
                for human_method in skill_human_scores.keys():
                    detailed_scores = [f"{skill_human_scores[human_method][tag]:.2f}" for tag in tag_groups[tag_group]]
                    detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
                    model_scores = f"{args.gen_model} {human_method:25} " + detailed_scores
                    print(model_scores)
            elif score_name == 'metric':
                for metric_method in skill_vqascores.keys():
                    detailed_scores = [f"{skill_vqascores[metric_method][tag]:.2f}" for tag in tag_groups[tag_group]]
                    detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
                    model_scores = f"{args.gen_model} {metric_method:25} " + detailed_scores
                    print(model_scores)
            print()
        print()
        

def show_ranking_performance(args, scores, score_name='accuracy'):
    # assert skill_vqascores[0].keys() == skill_human_scores[0].keys()
    tag_groups = {
    'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
    'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced'],
    'overall': ['basic', 'advanced', 'all']}
    for tag_group in tag_groups:
        # print vqascores and human scores in a table
        print(f"Tag Group: {tag_group} ({score_name} performance)")
        tag_header = f"{'Model':<30}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
        print(tag_header)
        detailed_scores = [f"{scores[tag]:.2f}" for tag in tag_groups[tag_group]]
        detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
        model_scores = f"{args.gen_model:<30}" + detailed_scores
        print(model_scores)
        print()
    print()

def show_top1_acc(args, top1_acc):
    tag_groups = {'overall': ['basic', 'advanced', 'all']}
    acc_methods = ['rerank_by_9', 'rerank_by_3']
    for tag_group in tag_groups:
        print(f"Tag Group: {tag_group} (Top 1 Accuracy)")
        tag_header = f"{'Model':<30}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
        print(tag_header)
        for i in range(len(acc_methods)):
            detailed_scores = [f"{top1_acc[i][tag]:.2f}" for tag in tag_groups[tag_group]]
            detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
            model_scores = f"{args.gen_model} {acc_methods[i]:25} " + detailed_scores
            print(model_scores)
        print()
    print()

def show_win_rate(args, win_rates):
    win_groups = {'overall': ['win', 'tie', 'lose']}
    # win_methods = win_rates.keys()
    for win_group in win_groups:
        print(f"Win Rate: {win_group}")
        win_header = f"{'Model (VS random DALLE3)':<30}" + " ".join([f"{tag:<20}" for tag in win_groups[win_group]])
        print(win_header)
        for win_method in win_rates.keys():
            detailed_scores = [f"{win_rates[win_method][tag]:.2f}" for tag in win_groups[win_group]]
            detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
            model_scores = f"{args.gen_model} {win_method:25} " + detailed_scores
            print(model_scores)
        print()

def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
        
    result_dir = f"{args.result_dir}/{args.gen_model}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    dataset = GenAIBench_Ranking(gen_model=args.gen_model, root_dir=args.root_dir)

    print(f"{args.gen_model} ranking dataset loaded successfully.")
    print(f"Dataset size: {len(dataset)}")
    result_path = f"{result_dir}/{args.model}.pt"

    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists. Skipping.")
        scores = torch.load(result_path)
    else:
        print(f"Computing scores for {args.model}.")
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

    # ------------------- Load skill tags -------------------
    images_to_prompt_idx = getattr(dataset, 'images_to_prompt_idx')
    prompt_num = int(len(dataset) / 9)
    tag_file = os.path.join(dataset.root_dir, 'genai_skills.json')
    tags = json.load(open(tag_file))
    tags.update({'all': images_to_prompt_idx})

    # ------------------- Compute Rerank idx (by 9) -------------------
    scores_per_prompt = torch.reshape(scores, (int(len(dataset) / 9), 9))  # [800, 9]
    our_scores_random = torch.mean(scores_per_prompt, dim=1)  # [800]

    rerank_idx_by_9 = torch.argmax(scores_per_prompt, dim=1)  # [800]
    our_scores_rerank = torch.max(scores_per_prompt, dim=1).values
    our_scores = {'random': our_scores_random, 'rerank_by_9': our_scores_rerank}

    # ------------------- Compute Rerank idx (by 3) -------------------
    rerank_idx_by_3 = []
    for prompt_idx in range(prompt_num):
        idx_img = []
        for img_idx in range(0,9,3):
            vqa_top_idx = torch.argmax(scores_per_prompt[prompt_idx][img_idx:img_idx+3])
            idx_img.append(vqa_top_idx+img_idx)
        rerank_idx_by_3.append(idx_img)
    rerank_idx_by_3 = torch.tensor(rerank_idx_by_3)  # [800, 3]

    # ------------------- Load and rerank human scores -------------------
    # human_scores keys: random, rerank_by_9, rerank_by_3, human_oracle_by9, human_oracle_by3, shape: [800]
    # human_scores_per_prompt shape: [800, 9]
    human_scores, human_scores_per_prompt = rerank_human_scores(dataset, rerank_idx_by_9, rerank_idx_by_3)

    # ------------------- Compute Scores on each skill-------------------
    metric_scores_skill = {}
    human_scores_skill = {}
    for key in our_scores.keys():
        metric_scores_skill.update({key: compute_scores_per_skill(our_scores[key], tags, images_to_prompt_idx)})
    for key in human_scores.keys():
        human_scores_skill.update({key: compute_scores_per_skill(human_scores[key], tags, images_to_prompt_idx)})
    
    show_performance(args, metric_scores_skill, human_scores_skill)

    # ------------------- Top 1 Accuracy -------------------
    top1_acc_rerank = compute_top1_acc_skill(scores_per_prompt, human_scores_per_prompt, tags, images_to_prompt_idx)
    top1_acc_rerank_group = compute_top1_acc_group_skill(scores_per_prompt, human_scores_per_prompt, tags, images_to_prompt_idx)
    top1_acc =[top1_acc_rerank, top1_acc_rerank_group]
    print("Metric model:", args.model)
    show_top1_acc(args, top1_acc)

    # ------------------- Pairwise Performance -------------------
    results = dataset.evaluate_scores(scores)
    ranking_results = compute_scores_per_skill(results['ranking_accuracy'], tags, images_to_prompt_idx)
    show_ranking_performance(args, ranking_results)

if __name__ == "__main__":
    main()
