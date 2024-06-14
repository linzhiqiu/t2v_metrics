# Evaluate on all datasets in VQAScore paper
# python gpt4_eval.py --model gpt-4-turbo
# python gpt4_eval.py --model gpt-4o
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
    parser.add_argument("--model", default="gpt-4-turbo", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--openai_key", default=None, type=str)
    parser.add_argument("--openai_key_path", default='./_OPENAI_API_KEY.txt', type=str)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--detail", type=str, default='auto', choices=['low', 'auto', 'high'])
    return parser.parse_args()


def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    assert not (args.openai_key is None and args.openai_key_path is None), "Please provide either openai_key or openai_key_path."
    if args.openai_key is None:
        args.openai_key = open(args.openai_key_path, 'r').read().strip()
    
    score_func = t2v_metrics.get_score_model(
        model=args.model, device=args.device, cache_dir=args.cache_dir, openai_key=args.openai_key, top_logprobs=args.top_logprobs)

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
        # EqBen_Mini,
        # TIFA160_DSG,
        # Pickapic_v1,
        # SeeTrue,
        # StanfordT23D,
        # T2VScore,
        # Flickr8K_CF,
    ]:
        print(f"Evaluating on {dataset_cls.__name__}.")
        dataset = dataset_cls(root_dir=args.root_dir)
        # check file size under 15mb
        for item in dataset:
            images = item['images']
            for image in images:
                assert os.path.getsize(image) < 15 * 1024 * 1024, f"File size of {image} is {os.path.getsize(image)/1048576} bytes, which is larger than 15mb."
                img_type = image.split('.')[-1]
                assert img_type in ['png', 'jpeg', 'jpg', 'gif', 'webp'], f"Image type {img_type} is not supported."
        scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
        score_path = f'./{args.model}_{args.detail}_{dataset_cls.__name__}.pt'
        # score_path = f'./{args.model}_{args.detail}_{dataset_cls.__name__}_no.pt' # when answer is "No"
        # score_path = f'./{args.model}_{args.detail}_{dataset_cls.__name__}_does_this_image_show_yes_or_no.pt' 
        # score_path = f'./{args.model}_{args.detail}_{dataset_cls.__name__}_yes_or_no.pt' 
        # score_path = f'./{args.model}_{args.detail}_{dataset_cls.__name__}_does_this_figure_show.pt' 
        import torch
        torch.save(scores, score_path)
        dataset.evaluate_scores(scores)

if __name__ == "__main__":
    main()

