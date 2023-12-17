# Evaluate on all dataset using a specific score

import argparse
import os
import t2i_metrics
from dataset import Winoground, EqBen_Mini, TIFA160_DSG, SugarCrepe


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2i_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model", default="clip-flant5-xxl", type=str)
    return parser.parse_args()


def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    score_func = t2i_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

    print(f"Performance of {args.model}.")
    for dataset_cls in [
        Winoground,
        EqBen_Mini,
        TIFA160_DSG,
        # SugarCrepe
    ]:
        dataset = dataset_cls(root_dir=args.root_dir)
        scores = score_func.batch_forward(dataset, batch_size=args.batch_size).cpu()
        dataset.evaluate_scores(scores)

if __name__ == "__main__":
    main()

