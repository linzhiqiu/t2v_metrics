# Evaluate on all dataset using a specific score
# python t2vscore_eval.py --model "CLIP Score"
import argparse
import os
import t2i_metrics
from dataset import T2VScore


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
    
    if args.model in ["CLIP Score","X-CLIP Score","BLIP-BLEU","T2VScore-A (GPT-4V)"]:
        print(f"Evaluating {args.model} on {T2VScore.__name__}")
        dataset = T2VScore(root_dir=args.root_dir)
        scores = dataset.get_scores_from_author(model=args.model)
        results = dataset.evaluate_scores(scores)
        return
    score_func = t2i_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

    kwargs = {}
    if args.question is not None:
        print(f"Using question template: {args.question}")
        kwargs['question_template'] = args.question
    if args.answer is not None:
        print(f"Using answer template: {args.answer}")
        kwargs['answer_template'] = args.answer
    
    os.makedirs("t2vscore_results", exist_ok=True)
    
    
    print(f"Performance of {args.model}.")
    for eval_mode in [
                    #   'first_frame',
                    #   'last_frame',
                    #   'sample_4_frame',
                      'grid_2_x_2',
                    #   'avg_frames'
                      ]:
        for dataset_cls in [
            T2VScore
        ]:  
            print(f"Evaluating {eval_mode} on {dataset_cls.__name__}")
            dataset = dataset_cls(root_dir=args.root_dir, eval_mode=eval_mode)
            scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
            results = dataset.evaluate_scores(scores)
            # dataset.save_results(scores, save_path=f"t2vscore_results/{args.model}_{eval_mode}.json")
            

if __name__ == "__main__":
    main()
