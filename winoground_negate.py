# Evaluate on all dataset using a specific score
# sleep 5000; CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model clip-flant5-xxl --answer "No"
# sleep 6000; CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model llava-v1.5-13b --answer "No"
# sleep 7000; CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model instructblip-flant5-xxl --answer "no"


# CUDA_VISIBLE_DEVICES=1 python winoground.py --model clip-flant5-xxl --question 'Does this figure show "{}"?' --answer "Yes it does"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model clip-flant5-xxl --question 'Does this figure show "{}"?' --answer "No it does not"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model clip-flant5-xxl --question 'Does this figure show "{}"? Please answer correct or wrong.' --answer "Wrong"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model clip-flant5-xxl --question 'Does this figure show "{}"? Please answer true or false.' --answer "False"

# sleep 4000; CUDA_VISIBLE_DEVICES=1 python winoground.py --model llava-v1.5-13b --question 'Does this figure show "{}"?' --answer "Yes it does"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model llava-v1.5-13b --question 'Does this figure show "{}"?' --answer "No it does not"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model llava-v1.5-13b --question 'Does this figure show "{}"? Please answer correct or wrong.' --answer "Wrong"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model llava-v1.5-13b --question 'Does this figure show "{}"? Please answer true or false.' --answer "False"

# sleep 8000; CUDA_VISIBLE_DEVICES=1 python winoground.py --model instructblip-flant5-xxl --question 'Question: Does this figure show "{}"?' --answer "yes it does"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model instructblip-flant5-xxl --question 'Question: Does this figure show "{}"?' --answer "no it does not"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model instructblip-flant5-xxl --question 'Question: Does this figure show "{}"? Please answer correct or wrong.' --answer "wrong"
# CUDA_VISIBLE_DEVICES=1 python winoground_negate.py --model instructblip-flant5-xxl --question 'Question: Does this figure show "{}"? Please answer true or false.' --answer "false"


import argparse
import os
import t2i_metrics
from dataset import Winoground, EqBen_Mini, TIFA160_DSG, Flickr8K_CF, Flickr8K_Expert, SeeTrue, NaturalBench, Stanford3D, MyStanford3D, EvalCrafter, GenAIBench_Image, Pickapic_v1_selected


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2i_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
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
    
    print(f"Performance of {args.model}.")
    for dataset_cls in [
        # Stanford3D,
        # MyStanford3D,
        # EvalCrafter,
        # NaturalBench,
        Winoground,
        EqBen_Mini,
        # SeeTrue,
        # TIFA160_DSG,
        # Flickr8K_Expert,
        # Flickr8K_CF,
        # GenAIBench_Image,
        # Pickapic_v1_selected,
    ]:
        dataset = dataset_cls(root_dir=args.root_dir)
        scores = -score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
        import torch
        torch.save(scores, f"winoground_results/{args.model}_{dataset_cls.__name__}_negate.pt")
        results = dataset.evaluate_scores(scores)

if __name__ == "__main__":
    main()

