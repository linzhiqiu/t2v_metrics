# Generate your own model's output on GenAI-Bench-Image (with 1600 prompt)
# Example scripts to run:
# python genai_bench/generate.py --model runwayml/stable-diffusion-v1-5

import argparse
import json
import os
import t2v_metrics
from dataset import GenAIBench_Image

import torch
import numpy as np
from tqdm import tqdm
from diffusers import DiffusionPipeline, StableDiffusionPipeline


torch.set_grad_enabled(False)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num_prompts", default=1600, type=int, choices=[527, 1600])
    parser.add_argument(
        "--gen_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Huggingface model name (You will need to modify the scripe to use a different model)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="dir to write results to",
        default="./outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    os.makedirs(os.path.join(args.output_dir, args.gen_model), exist_ok=True)

    dataset =  GenAIBench_Image(root_dir=args.root_dir, num_prompts=args.num_prompts) # 'num_prompts' is the number of prompts in GenAI-Bench（1600 in GenAI-Bench paper; 527 in VQAScore paper）
    
    # Load model
    if args.gen_model == "stabilityai/stable-diffusion-xl-base-1.0":
        model = DiffusionPipeline.from_pretrained(args.gen_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        model.enable_xformers_memory_efficient_attention()
    else:
        model = StableDiffusionPipeline.from_pretrained(args.gen_model, torch_dtype=torch.float16)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.enable_attention_slicing()

    torch.manual_seed(args.seed)

    with torch.no_grad():
        for prompt_idx in tqdm(dataset.dataset.keys()):
            prompt = dataset.dataset[prompt_idx]['prompt']
            # print(f"Prompt: {prompt}")
            # Generate images
            sample = model(
                prompt,
                height=args.H,
                width=args.W,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                num_images_per_prompt=1,
                negative_prompt=args.negative_prompt or None
            ).images[0]
            sample.save(os.path.join(args.output_dir, args.gen_model, f"{prompt_idx}.jpeg"))

    print(f"Done, saved to {os.path.join(args.output_dir, args.gen_model)}")
    print(f"Please run python genai_bench/evaluate.py --output_dir {args.output_dir} --gen_model {args.gen_model} to evaluate the generated images.")


if __name__ == "__main__":
    args = config()
    main(args)