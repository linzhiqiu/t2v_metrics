# Evaluate on all dataset using a specific score
# CUDA_VISIBLE_DEVICES=3 python stanfordt23d_eval.py --model openai:ViT-L-14-336
# CUDA_VISIBLE_DEVICES=3 python stanfordt23d_eval.py --model openai:ViT-B-32
# CUDA_VISIBLE_DEVICES=4 python stanfordt23d_eval.py --model clip-flant5-xxl
# CUDA_VISIBLE_DEVICES=5 python stanfordt23d_eval.py --model llava-v1.5-13b
# CUDA_VISIBLE_DEVICES=6 python stanfordt23d_eval.py --model instructblip-flant5-xxl
# CUDA_VISIBLE_DEVICES=7 python stanfordt23d_eval.py --model image-reward-v1
# CUDA_VISIBLE_DEVICES=2 python stanfordt23d_eval.py --model pickscore-v1
# CUDA_VISIBLE_DEVICES=1 python stanfordt23d_eval.py --model hpsv2
# CUDA_VISIBLE_DEVICES=1 python stanfordt23d_eval.py --model blip2-itm
import argparse
import os
import t2i_metrics
from dataset import StanfordT23D
import json

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
    
    score_func = t2i_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

    kwargs = {}
    if args.question is not None:
        print(f"Using question template: {args.question}")
        kwargs['question_template'] = args.question
    if args.answer is not None:
        print(f"Using answer template: {args.answer}")
        kwargs['answer_template'] = args.answer
    
    os.makedirs("stanfordt23d_results", exist_ok=True)
    
    print(f"Performance of {args.model}.")
    all_results = {}
    for eval_mode in [
                    #   'first_rgb_view',
                    #   'last_rgb_view',
                    #   'sample_4_rgb_views',
                    #   'sample_4_normal_views',
                    #   'sample_9_rgb_views',
                    #   'sample_9_normal_views',
                    #   'rgb_grid_2_x_2',
                    #   'rgb_grid_3_x_3',
                      'rgb_views'
                      ]:
        for dataset_cls in [
            StanfordT23D
        ]:  
            print(f"Evaluating {eval_mode} on {dataset_cls.__name__}")
            dataset = dataset_cls(root_dir=args.root_dir, eval_mode=eval_mode)
            scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
            results = dataset.evaluate_scores(scores)
            all_results[eval_mode] = results
            # dataset.save_results(scores, save_path=f"stanfordt23d_results/{args.model}_{eval_mode}.json")
    # if not os.path.exists("stanfordt23d_results"):
    #     os.makedirs("stanfordt23d_results")
    # option_str = "" if args.question is None and args.answer is None else f"_{args.question}_{args.answer}"
    # json.dump(all_results, open(f"stanfordt23d_results/{args.model}{option_str}.json", "w"), indent=4)

if __name__ == "__main__":
    main()
