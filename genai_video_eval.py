# Evaluate on GenAI-Bench-Video using a specific model
# Example scripts to run:
# VQAScore: python genai_video_eval.py --model clip-flant5-xxl
# CLIPScore: python genai_video_eval.py --model openai:ViT-L-14-336
import argparse
import os
import t2v_metrics
from dataset import GenAIBench_Video
import json
import torch
import numpy as np
from genai_image_eval import show_performance_per_skill

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_prompts", default=800, type=int, choices=[527, 800])
    parser.add_argument("--model", default="clip-flant5-xxl", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--result_dir", default="./genai_video_results", type=str)
    parser.add_argument("--eval_mode", default="avg_frames", type=str)
    return parser.parse_args()


def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = f"{args.result_dir}/{args.model}_{args.eval_mode}_{args.num_prompts}_prompts.pt"
    dataset = GenAIBench_Video(root_dir=args.root_dir, eval_mode=args.eval_mode, num_prompts=args.num_prompts)
    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists. Skipping.")
        scores = torch.load(result_path)
    else:
        score_func = t2v_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

        kwargs = {}
        if args.question is not None:
            print(f"Using question template: {args.question}")
            kwargs['question_template'] = args.question
        if args.answer is not None:
            print(f"Using answer template: {args.answer}")
            kwargs['answer_template'] = args.answer
        
        print(f"Performance of {args.model} on using {args.eval_mode}.")
        scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
        torch.save(scores, result_path)
    
    ### Get performance per skill
    our_scores = scores.mean(axis=1)
    show_performance_per_skill(our_scores, dataset, items_name='videos', prompt_to_items_name='prompt_to_videos', print_std=True)
    
    print("Alignment Performance")
    ### Alignment performance
    dataset.evaluate_scores(scores)
    
    

if __name__ == "__main__":
    main()
