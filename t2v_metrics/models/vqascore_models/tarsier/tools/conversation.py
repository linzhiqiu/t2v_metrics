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

# copy and modify from: https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/conversation.py
from PIL import Image
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from dataset.custom_data_parsers.utils import put_pred_to_data_dict, get_prompt_from_data_dict
from dataset.tarsier_datamodule import TarsierDataProcessor
from dataset.utils import *

from enum import auto, Enum
import os
import re

data_dict_tmp = {
    "messages": [
        {
            "role": "user", 
            "content": [
                {
                    "type": "video", 
                    "video": {
                        "video_file": "/mnt/hdfs/vlm/videos/movies_aligned_0523/tt8266310/tt8266310_1.50.24-1.50.29.mp4"}
                },
                {
                    "type": "text", 
                    "text": "Describe the video in detail."
                }
            ]
        }, 
        {
            "role": "assistant", 
            "content": [
                {
                    "type": "text", 
                    "text": "A man in the driver's seat, wearing a black jacket with a maroon shirt, fastens his seatbelt while smiling at the man in the passenger seat, who is adjusting his position. The passenger, also wearing a black jacket with a maroon shirt, turns to look forward and smiles. The driver then leans forward to start the car and leans back in his seat. In the background, a beige car is visible through the window."
            }]}
    ], 
    "dataset": "video_caption", 
    "task": "video/caption", 
    "idx": 0, 
}


IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

def get_data_dict(conv, max_n_frames=None):
    data_dict = {
        "messages": []
    }
    for i, (role, message) in enumerate(conv.messages):
        if message:
            text = message["text"]
            content_type = message["type"]
            content = {}
            if content_type == "text":
                content['type'] = 'text'
                content['text'] = text
                task = "text-only"
            elif content_type == "video":
                content['type'] = 'video'
                content['video'] = {
                    "video_file": text
                }
                if max_n_frames is not None:
                    content['video']['n_frames'] = max_n_frames
                task = "video/QA"
            elif content_type == "image":
                content['type'] = 'image'
                content['image'] = {
                    "image_file": text
                }
                task = "image/QA"
            else:
                content['type'] = 'text'
                content['text'] = text
                task = "text-only"
            if data_dict['messages'] and data_dict['messages'][-1]['role'] == role:
                data_dict['messages'][-1]['content'].append(content)
            else:
                data_dict['messages'].append({
                    "role": role,
                    "content": [content]
                })
    data_dict['dataset'] = task
    data_dict['task'] = task
    check_data_format(data_dict)
    return data_dict


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Chat:
    def __init__(self, model, processor: TarsierDataProcessor, device='cuda', debug=False):
        self.model = model
        self.processor = processor
        self.device = device
        self.debug = debug
        stop_words_ids = [torch.tensor([self.processor.processor.tokenizer.eos_token_id]).to(device)]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self,text,conv):
        conv.messages.append([conv.roles[0], {"text": text, "type": "text"}])
        return conv

    def prepare_model_inputs(self, conv, n_frames=None):
        # print(conv.messages)
        data_dict = get_data_dict(conv, n_frames)
        if self.debug:
            # print(f"visual_data_file: {visual_data_file}", flush=True)
            print(f"###Prompt:\n{get_prompt_from_data_dict(data_dict)}")

        batch_data = self.processor(data_dict)
        model_inputs = {}
        for k, v in batch_data.items():
            if not isinstance(v, torch.Tensor):
                continue
            model_inputs[k] = v.to(self.device)
        return model_inputs, conv

    def answer(self, conv, n_frames=None, max_new_tokens=256, num_beams=1, min_length=1, top_p=1.0,
               repetition_penalty=1.0, length_penalty=1, temperature=0):
        inputs, conv = self.prepare_model_inputs(conv, n_frames)
        if self.model is not None:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=True if temperature > 0 else False,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
            output_text = self.processor.processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
        else:
            output_text = "Fake respone as launched in debug mode!"
        conv.messages.append(
            [conv.roles[1], {"text": output_text, "type": "text"}]
        )
        return output_text, conv

class EasyDict(dict):
    """
    Get attributes

    >>> d = EasyDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'bar'

    Works recursively

    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1
    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, d)

conv_tarsier = EasyDict({
    "system": "",
    "roles": ("USER", "ASSISTANT"),
    "messages": [],
    "sep1": " ",
    "sep2": "</s>",
}
)

conv_tarsier_yi = EasyDict({
    "system": "",
    "roles": ("USER", "ASSISTANT"),
    "messages": [],
    "sep1": " ",
    "sep2": "<|endoftext|>",
}
)

conv_tarsier_qwen2_vl = EasyDict({
    "system": "",
    "roles": ("user", "assistant"),
    "messages": [],
}
)

conv_templates = {
    "tarsier2-7b": conv_tarsier_qwen2_vl
}
