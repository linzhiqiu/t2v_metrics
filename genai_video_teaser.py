# Evaluate on all dataset using a specific score
# CUDA_VISIBLE_DEVICES=1 python genai_video_eval.py --model clip-flant5-xxl
# CUDA_VISIBLE_DEVICES=2 python genai_video_eval.py --model llava-v1.5-13b
# CUDA_VISIBLE_DEVICES=3 python genai_video_eval.py --model instructblip-flant5-xxl
# CUDA_VISIBLE_DEVICES=4 python genai_video_eval.py --model image-reward-v1
# CUDA_VISIBLE_DEVICES=5 python genai_video_eval.py --model pickscore-v1
# CUDA_VISIBLE_DEVICES=5 python genai_video_eval.py --model hpsv2
# CUDA_VISIBLE_DEVICES=5 python genai_video_eval.py --model blip2-itm
# CUDA_VISIBLE_DEVICES=5 python genai_video_eval.py --model openai:ViT-L-14-336
# (skip) CUDA_VISIBLE_DEVICES=5 python genai_video_eval.py --model openai:ViT-B-32
import argparse
import os
import t2i_metrics
from dataset import GenAIBench_Video
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
    
    result_dir = "genai_video_results"
    os.makedirs(result_dir, exist_ok=True)
    tag_result = {}
    clip_flant5_result_path = f"{result_dir}/clip-flant5-xxl_avg_frames_all_528.pt"
    llava_result_path = f"{result_dir}/llava-v1.5-13b_avg_frames_all_528.pt"
    assert os.path.exists(clip_flant5_result_path) and os.path.exists(llava_result_path)
    vqa_scores = torch.load(clip_flant5_result_path)
    llava_scores = torch.load(llava_result_path)
    vqa_scores = vqa_scores.mean(axis=1)
    llava_scores = llava_scores.mean(axis=1)
    import numpy as np
    dataset = GenAIBench_Video(root_dir=args.root_dir, filename='genai_video_final')
    human_scores = [np.array(dataset.videos[idx]['human_alignment']).mean() for idx in range(len(dataset.videos))]
    videos = {}
    for prompt_idx in dataset.dataset:
        for video_idx in dataset.prompt_to_videos[prompt_idx]:
            model = dataset.videos[video_idx]['model']
            if model not in videos:
                videos[model] = {}
            videos[model][prompt_idx] = video_idx
    
    new_dataset = {}
    keys_to_delete = ['Zeroscope']
    for key in keys_to_delete:
        del videos[key]
    for prompt_idx in dataset.dataset:
        new_dataset[prompt_idx] = dataset.dataset[prompt_idx]
        for key in keys_to_delete:
            del new_dataset[prompt_idx]['models'][key]
        clip_flant5_scores = [float(vqa_scores[videos[model][prompt_idx]]) for model in videos]
        llavascores = [float(llava_scores[videos[model][prompt_idx]]) for model in videos]
        humanscores = [human_scores[videos[model][prompt_idx]] for model in videos]
        clip_flant5_rankings = np.argsort(clip_flant5_scores)[::-1]
        llava_rankings = np.argsort(llavascores)[::-1]
        human_rankings = np.argsort(humanscores)[::-1]
        
        # only save if clip_flant5_rankings does not agrees with human
        if clip_flant5_rankings.tolist() == human_rankings.tolist():
            del new_dataset[prompt_idx]
            continue
        
        for model_idx, model in enumerate(videos):
            new_dataset[prompt_idx]['models'][model] = {
                'clip_flant5': f"{float(vqa_scores[videos[model][prompt_idx]]):.3f}",
                'llava': f"{float(llava_scores[videos[model][prompt_idx]]):.3f}",
                'human': f"{human_scores[videos[model][prompt_idx]]:.2f}",
                'clip_flant5_rank': f"{clip_flant5_rankings.tolist().index(model_idx) + 1}",
                'llava_rank': f"{llava_rankings.tolist().index(model_idx) + 1}",
                'human_rank': f"{human_rankings.tolist().index(model_idx) + 1}",
                # 'human_all': dataset.dataset[prompt_idx]['models'][model],
            }
    # Sort prompt_idx by the average clip_flant5 - llava across all models
    new_dataset = dict(sorted(new_dataset.items(), key=lambda item: np.mean([float(item[1]['models'][model]['clip_flant5']) - float(item[1]['models'][model]['llava']) for model in item[1]['models']]), reverse=True))
    
    with open("genai_video_teaser.json", "w") as f:
        json.dump(new_dataset, f, indent=4)


if __name__ == "__main__":
    main()
