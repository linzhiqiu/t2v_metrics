# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tasks.utils import load_model_and_processor
from dataset.custom_data_parsers.utils import put_pred_to_data_dict, get_prompt_from_data_dict
from dataset.utils import *

import os
import torch
from tqdm import tqdm
import yaml

def process_one(model, processor, prompt, video_file, generate_kwargs):
    # inputs = processor(prompt, video_file, edit_prompt=True, return_prompt=True)
    sample = format_one_sample(video_file, prompt)
    batch_data = processor(sample)
    print(f"###Prompt:\n{get_prompt_from_data_dict(sample)}")
    model_inputs = {}
    for k, v in batch_data.items():
        if not isinstance(v, torch.Tensor):
            continue
        model_inputs[k] = v.to(model.device)
    outputs = model.generate(
        **model_inputs,
        **generate_kwargs,
    )
    # print(processor.processor.tokenizer.decode(outputs[0][:model_inputs['input_ids'][0].shape[0]], skip_special_tokens=True))
    output_text = processor.processor.tokenizer.decode(outputs[0][model_inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
    return output_text

def run():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--config', type=str, default="configs/tarser2_default_config.yaml")
    parser.add_argument('--instruction', type=str, default="Describe the video in detail.", help='Input prompt.')
    parser.add_argument('--input_path', type=str, default="assets/examples", help='Path to video/image; or Dir to videos/images')
    # parser.add_argument("--max_n_frames", type=int, default=16, help="Max number of frames to apply average sampling from the given video.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="max number of generated tokens")
    parser.add_argument("--top_p", type=float, default=1, help="Top_p sampling")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature > 0 to enable sampling generation.")

    args = parser.parse_args()

    # model, processor = load_model_and_processor(args.model_name_or_path, max_n_frames=args.max_n_frames) # max_n_frames set in config_file
    data_config = yaml.safe_load(open(args.config, 'r'))
    model, processor = load_model_and_processor(args.model_name_or_path, data_config=data_config)
    
    generate_kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "use_cache": True
    }
    assert os.path.exists(args.input_path), f"input_path not exist: {args.input_path}"
    if os.path.isdir(args.input_path):
        input_files = [os.path.join(args.input_path, fn) for fn in os.listdir(args.input_path) if get_visual_type(fn) in ['video', 'gif', 'image']]
    elif get_visual_type(args.input_path) in ['video', 'gif', 'image']:
        input_files = [args.input_path]
    assert len(input_files) > 0, f"None valid input file in: {args.input_path} {VALID_DATA_FORMAT_STRING}"

    for input_file in tqdm(input_files, desc="Generating..."):
        visual_type = get_visual_type(input_file)
        if args.instruction:
            prompt = args.instruction
        else:
            if visual_type == 'image':
                prompt = "Describe the image in detail."
            else:
                prompt = "Describe the video in detail."
        
        pred = process_one(model, processor, prompt, input_file, generate_kwargs)
        print(f"###Prediction:\n{pred}")
        print('-'*100)

        
if __name__ == "__main__":
    # python3 -m tasks.inference_quick_start --model_name_or_path /tmp/tarsier2-1226-dpo --config configs/tarser2_default_config.yaml --input_path /mnt/bn/videonasi18n/wangjw/workspace/tarsier/diving.mp4 --instruction "List the names of all sponsors on the background wall."
    run()
