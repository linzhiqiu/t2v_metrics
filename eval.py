# Evaluate on all datasets in VQAScore paper

import argparse
import os
import t2v_metrics
from dataset import Winoground, EqBen_Mini, StanfordT23D, TIFA160_DSG, Flickr8K_CF, SeeTrue, Pickapic_v1, T2VScore


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str) 
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
    
    score_func = t2v_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

    kwargs = {}
    if args.question is not None:
        print(f"Using question template: {args.question}")
        kwargs['question_template'] = args.question
    if args.answer is not None:
        print(f"Using answer template: {args.answer}")
        kwargs['answer_template'] = args.answer
    
    print(f"Performance of {args.model}.")
    for dataset_cls in [
        Winoground,
        EqBen_Mini,
        TIFA160_DSG,
        Pickapic_v1,
        SeeTrue,
        StanfordT23D,
        T2VScore,
        Flickr8K_CF,
    ]:
        
        dataset = dataset_cls(root_dir=args.root_dir)
        scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
        dataset.evaluate_scores(scores)

if __name__ == "__main__":
    main()

