import cv2
import base64
import numpy as np
from typing import List, Union
import torch
from openai import OpenAI
import tiktoken
import os

from .vqa_model import VQAScoreModel

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = 'Yes'

GPT4V_MODELS = {
    'gpt-4-turbo': {},
    'gpt-4o': {},
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_type(image_path):
    image_type = image_path.split('.')[-1].lower()
    assert image_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']
    return image_type

def extract_frames(video_path, num_frames):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // num_frames
    frames = []
    
    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = video.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            frames.append(base64_frame)
    
    video.release()
    return frames

class GPT4VModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='gpt-4-turbo',
                 device='cuda',
                 cache_dir=None,
                 openai_key=None,
                 top_logprobs=2):
        assert model_name in GPT4V_MODELS
        assert openai_key is not None, "Please provide an OpenAI API key"
        self.openai_key = openai_key
        self.top_logprobs = top_logprobs
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.client = OpenAI(api_key=self.openai_key)

    def load_images(self, paths: List[str], num_frames: int = None) -> List[dict]:
        loaded_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                frames = extract_frames(path, num_frames)
                loaded_data.append({
                    'path': path,
                    'type': 'video',
                    'frames': frames
                })
            else:  # Image file
                loaded_data.append({
                    'path': path,
                    'type': get_image_type(path),
                    'base64': encode_image(path)
                })
        return loaded_data

    def forward_single(self, data, question):#, answer):
        try:
            if data['type'] == 'video':
                content = [
                    {"type": "text", "text": question},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame}"}} for frame in data['frames']]
                ]
            else:
                content = [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/{data['type']};base64,{data['base64']}"}}
                ]

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                logprobs=True,
                top_logprobs=self.top_logprobs,
            )

        except:
            try: # Second try
                if data['type'] == 'video':
                    content = [
                        {"type": "text", "text": question},
                        *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame}"}} for frame in data['frames']]
                    ]
                else:
                    content = [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/{data['type']};base64,{data['base64']}"}}
                    ]

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                )
            except Exception as e: # Old Error Handling
                print(f"Failed image: {data['path']} and question: {question} and answer: {answer}")
                print(f"Error: {str(e)}")
                return torch.Tensor([0.0])

                
        is_generated = False
        for top_logprob in completion.choices[0].logprobs.content[0].top_logprobs:
            if top_logprob.token == 'Yes' or top_logprob.token == 'yes':
                is_generated = True
                return torch.Tensor([top_logprob.logprob]).exp()
            elif top_logprob.token == 'No' or top_logprob.token == 'no':
                is_generated = True
                return 1 - torch.Tensor([top_logprob.logprob]).exp()
        if not is_generated:
            print(f"Warning: 'Yes' not included in gpt4o log probs: {data['path']} and question: {question} and answer: {answer}")
            print(completion.choices[0].logprobs.content[0].top_logprobs)
            return torch.Tensor([0.0])

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = default_question_template,
                answer_template: str = default_answer_template,
                num_frames: int = 5) -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        for ans in answers:
            ans_tokens = self.tokenizer.encode(ans)
            assert len(ans_tokens) == 1, "Currently only support single token answers"

        loaded_data = self.load_images(paths, num_frames)

        lm_prob = torch.zeros(len(paths))

        for idx, (data, question, answer) in enumerate(zip(loaded_data, questions, answers)):
            lm_prob[idx] = self.forward_single(data, question)#, answer)

        return lm_prob
    
    def generate_single(self, data, question):
        try:
            if data['type'] == 'video':
                content = [
                    {"type": "text", "text": question},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame}"}} for frame in data['frames']]
                ]
            else:
                content = [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/{data['type']};base64,{data['base64']}"}}
                ]

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=256,
            )

        except:
            try:  # Second try
                if data['type'] == 'video':
                    content = [
                        {"type": "text", "text": question},
                        *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame}"}} for frame in data['frames']]
                    ]
                else:
                    content = [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/{data['type']};base64,{data['base64']}"}}
                    ]

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=256,
                )
            except Exception as e:
                print(f"Failed image: {data['path']} and question: {question}")
                print(f"Error: {str(e)}")
                return ""

        return completion.choices[0].message.content

    def generate(self,
            paths: List[str],
            texts: List[str],
            num_frames: int = 5) -> List[str]:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = texts
        loaded_data = self.load_images(paths, num_frames)

        generated_outputs = []

        for idx, (data, question) in enumerate(zip(loaded_data, questions)):
            generated_text = self.generate_single(data, question)
            generated_outputs.append(generated_text)

        return generated_outputs