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
import argparse
import os
import torch
from copy import deepcopy
from transformers import StoppingCriteriaList
from tasks.utils import load_model_and_processor
from dataset.utils import *
from tools.conversation import Chat, conv_templates, StoppingCriteriaSub
from transformers import TextStreamer
from tools.color import Color


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Load Model
    print(f"### Start loading model...")
    model, processor = load_model_and_processor(args.model_name_or_path, args.config)
    print(f"### Finish loading model.")
    if 'tarsier2' in args.model_name_or_path.lower():
        conv_type = 'tarsier2-7b'
    else:
        if '7b' in args.model_name_or_path.lower():
            conv_type = 'tarsier-7b'
        elif '13b' in args.model_name_or_path.lower():
            conv_type = 'tarsier-13b'
        elif '34b' in args.model_name_or_path.lower():
            conv_type = 'tarsier-34b'
        else:
            raise ValueError(f"Unknow model: {args.model_name_or_path}")

    chat = Chat(model, processor, device=device, debug = args.debug)
    conv = deepcopy(conv_templates[conv_type])

    img_path = ''
    has_img = False
    while True:
        if not has_img:
            try:
                img_path = input(Color.green(f"{conv.roles[1]}: ") +  "Input a file path of your image/video:")
                img_path = img_path.strip()
                if not (os.path.exists(img_path) and get_visual_type(img_path) in ['video', 'gif', 'image']):
                    continue
                has_img = True 
                conv.messages.append([conv.roles[0], {"type": "video", "text": img_path}])
                print(Color.green(f"{conv.roles[1]}: ") + "Received your file. Now let's start conversation! :)")
                print(Color.red(f"<Input \'exit\' to exit and \'reset\' to restart>"))
            except Exception as e:
                print(f"Error: {e}")
                print("exit...")
                exit()
        inp = ""
        while inp == "":
            inp = input(Color.blue(f"{conv.roles[0]}: ")).strip()
        if inp.strip() == 'exit':
            print("exit...")
            exit()
        elif inp.strip() == "reset":
            conv = deepcopy(conv_templates[conv_type])
            img_path = ''
            continue
        conv = chat.ask(inp, conv)

        stop_words_ids = [torch.tensor([processor.processor.tokenizer.eos_token_id]).to(device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        streamer = TextStreamer(processor.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        inputs, conv = chat.prepare_model_inputs(conv, args.max_n_frames)
        print("conv:", conv)
        print(Color.green(f"{conv.roles[1]}: "), end="")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = processor.processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
        conv.messages.append(
            [conv.roles[1], {"text": outputs, "type": "text"}]
        )

        if args.debug:
            print(f"Conversation state: {conv}")

if __name__ == "__main__":
    # python3 -m tasks.demo_cli --model_name_or_path /tmp/tarsier2-1226-dpo --config configs/tarser2_default_config.yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--config', type=str, default="configs/tarser2_default_config.yaml")
    parser.add_argument("--max_n_frames", type=int, default=16, help="Max number of frames to apply average sampling from the given video.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--top_p", type=float, default=1, help="Top_p sampling")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature > 0 to enable sampling generation.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
