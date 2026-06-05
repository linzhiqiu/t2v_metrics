import cv2
import base64
import numpy as np
from typing import List, Union
import torch
from openai import OpenAI
import tiktoken
import os

from .vqa_model import VQAScoreModel

default_question_template = 'Does this figure show "{}"? Please answer Yes or No.'
default_answer_template = 'Yes'

# Look into incorporating reasoning efforts later!

GPT4V_MODELS = {
    'gpt-4o':            {},
    'gpt-4.1':           {},
    # 'gpt-5':             {},
    # 'gpt-5.1':           {},
    # 'gpt-5.4':           {},
    # 'gpt-5.5':           {},
    # 'chat-latest':       {},
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
                api_key=None,
                top_logprobs=2):
        assert model_name in GPT4V_MODELS
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert api_key is not None, "No OpenAI API key provided. Pass api_key= or set the OPENAI_API_KEY environment variable."
        self.api_key = api_key
        self.top_logprobs = top_logprobs
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')
        self.client = OpenAI(api_key=self.api_key)

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
            elif isinstance(path, list):
                loaded_data.append({
                    'path': path,
                    'type': 'frame_list',
                    'frames': []
                })
            else:  # Image file
                loaded_data.append({
                    'path': path,
                    'type': get_image_type(path),
                    'base64': encode_image(path)
                })
        return loaded_data


    def forward_single(self, data, question, answer, max_new_tokens=1):
        """
        Forward pass for a single sample with flexible token extraction.
        
        Args:
            data: Image/video data dict (can be None for text-only)
            question: Question text
            answer: Expected answer (can be multi-token)
            max_new_tokens: Maximum tokens to generate (1 for binary, 100+ for CoT)
        """
        try:
            # Build content based on whether we have image/video data
            if data is None:
                # Text-only mode
                content = [{"type": "text", "text": question}]
            elif data['type'] == 'video':
                content = [
                    {"type": "text", "text": question},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame}"}} for frame in data['frames']]
                ]
            else:
                content = [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/{data['type']};base64,{data['base64']}"}}
                ]

            token_limit_key = 'max_completion_tokens' if 'gpt-5' in self.model_name else 'max_tokens'

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                logprobs=True,
                top_logprobs=self.top_logprobs,
                **{token_limit_key: max_new_tokens},
            )

        except Exception:
            try:  # Second try
                # Build content based on whether we have image/video data
                if data is None:
                    # Text-only mode
                    content = [{"type": "text", "text": question}]
                elif data['type'] == 'video':
                    content = [
                        {"type": "text", "text": question},
                        *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame}"}} for frame in data['frames']]
                    ]
                else:
                    content = [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/{data['type']};base64,{data['base64']}"}}
                    ]
                token_limit_key = 'max_completion_tokens' if 'gpt-5' in self.model_name else 'max_tokens'

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                    **{token_limit_key: max_new_tokens},
                )

            except Exception as e:
                path_info = data['path'] if data else 'text-only'
                print(f"Failed: {path_info} and question: {question} and answer: {answer}")
                print(f"Error: {str(e)}")
                return torch.Tensor([0.0])

        # Extract generated text and logprobs
        generated_text = completion.choices[0].message.content.strip()
        logprobs_content = completion.choices[0].logprobs.content
        
        # Extract from FIRST token (Yes/No is the first generated token)
        first_token_data = logprobs_content[0]
        first_token_text = first_token_data.token.strip()
        
        # DEBUG: Print generated output and verification
        # print(f"\n[GPT-4o] Generated: {generated_text}")
        # print(f"[GPT-4o] Total tokens: {len(logprobs_content)}, First token: '{first_token_text}'")
        # Extract probability from first token — scan all candidates first, then decide
        target   = answer.lower().strip()
        opposite = "no" if target == "yes" else "yes"

        target_logprob   = None
        opposite_logprob = None

        for top_logprob in first_token_data.top_logprobs:
            token = top_logprob.token.strip().lower()
            if token == target:
                target_logprob = top_logprob.logprob
            elif token == opposite:
                opposite_logprob = top_logprob.logprob

        if target_logprob is not None:
            return torch.Tensor([target_logprob]).exp()
        elif opposite_logprob is not None:
            return 1 - torch.Tensor([opposite_logprob]).exp()
        else:
            print(f"[GPT] Warning: neither '{target}' nor '{opposite}' in top {self.top_logprobs} logprobs")
            print(f"[GPT] Top logprobs: {[lp.token for lp in first_token_data.top_logprobs]}")
            return torch.Tensor([0.0])

    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str = default_question_template,
                answer_template: str = default_answer_template,
                num_frames: int = 4,
                fps: int=None,
                max_new_tokens: int = 1,
                temperature: float=None) -> torch.Tensor:
        """
        Forward pass with flexible token extraction.
        
        Args:
            images: List of image/video paths (can be None for text-only)
            texts: List of text prompts
            question_template: Template for questions
            answer_template: Template for answers
            num_frames: Number of frames for videos
            max_new_tokens: Max tokens to generate (1 for binary, 100+ for CoT)
        
        Returns:
            Tensor of token probabilities
        """
        # Handle text-only mode
        if images is None:
            loaded_data = [None] * len(texts)
        else:
            assert len(images) == len(texts), "Number of paths and texts must match"
            loaded_data = self.load_images(images, num_frames)
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        lm_prob = torch.zeros(len(texts))

        for idx, (data, question, answer) in enumerate(zip(loaded_data, questions, answers)):
            lm_prob[idx] = self.forward_single(data, question, answer, max_new_tokens=max_new_tokens)

        return lm_prob

    def generate_single(self, data, question, max_new_tokens):
        try:
            # Build content based on whether we have image/video data
            if data is None:
                # Text-only mode
                content = [{"type": "text", "text": question}]
            elif data['type'] == 'video':
                content = [
                    {"type": "text", "text": question},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame}"}} for frame in data['frames']]
                ]
            else:
                content = [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/{data['type']};base64,{data['base64']}"}}
                ]
            
            token_limit_key = 'max_completion_tokens' if 'gpt-5' in self.model_name else 'max_tokens'

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                **{token_limit_key: max_new_tokens},
            )

        except:
            try:  # Second try
                # Build content based on whether we have image/video data
                if data is None:
                    # Text-only mode
                    content = [{"type": "text", "text": question}]
                elif data['type'] == 'video':
                    content = [
                        {"type": "text", "text": question},
                        *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{frame}"}} for frame in data['frames']]
                    ]
                else:
                    content = [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/{data['type']};base64,{data['base64']}"}}
                    ]
                
                token_limit_key = 'max_completion_tokens' if 'gpt-5' in self.model_name else 'max_tokens'

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    **{token_limit_key: max_new_tokens},
                )
            except Exception as e:
                path_info = data['path'] if data else 'text-only'
                print(f"Failed: {path_info} and question: {question}")
                print(f"Error: {str(e)}")
                return ""

        return completion.choices[0].message.content

    def generate(self,
            images: List[str],
            texts: List[str],
            num_frames: int = 5,
            fps: int = None,
            max_new_tokens: int = 2048,
            temperature: float=None) -> List[str]:
        """
        Generate text responses.
        
        Args:
            images: List of image/video paths (can be None for text-only)
            texts: List of text prompts
            num_frames: Number of frames for videos
            max_new_tokens: Max tokens to generate
        
        Returns:
            List of generated text responses
        """
        # Handle text-only mode
        if images is None:
            loaded_data = [None] * len(texts)
        else:
            assert len(images) == len(texts), "Number of paths and texts must match"
            loaded_data = self.load_images(images, num_frames)
        
        questions = texts
        generated_outputs = []

        for idx, (data, question) in enumerate(zip(loaded_data, questions)):
            generated_text = self.generate_single(data, question, max_new_tokens)
            generated_outputs.append(generated_text)

        return generated_outputs