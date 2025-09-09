import cv2
import base64
import numpy as np
from typing import List, Union
import torch
import os
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
)

from .vqa_model import VQAScoreModel

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = 'Yes'

GEMINI_MODELS = {
    'gemini-1.5-pro': {},
    'gemini-1.5-flash': {},
    'gemini-2.5-flash': {},
    'gemini-2.5-pro': {},
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

def get_image_type(image_path):
    image_type = image_path.split('.')[-1].lower()
    assert image_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']
    return f"image/{image_type}"

# For extracting frames if needed, but Gemini supports direct video input
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
            frames.append(buffer.tobytes())
    
    video.release()
    return frames

class GeminiModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    
    def __init__(self,
                 model_name='gemini-2.5-pro-preview-03-25',
                 device='cuda',
                 cache_dir=None,
                 project_id=None,
                 api_key=None,
                 location=None,
                 vertex=False,
                 logprobs=5):
        assert model_name in GEMINI_MODELS, f"Model {model_name} not supported. Choose from {list(GEMINI_MODELS.keys())}"
        self.project_id = project_id
        self.location = location or "us-central1"
        self.logprobs = logprobs
        self.api_key = api_key
        if project_id:
            self.vertex = True
        else:
            self.vertex = False
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        """Initialize the Gemini client."""
        if self.vertex:
            self.client = genai.Client(vertexai=True, project=self.project_id, location="global")
        else:
            self.client = genai.Client(api_key=self.api_key)

    def load_images(self, paths: List[str], num_frames: int = None) -> List[dict]:
        loaded_data = []
        for path in paths:
            # Handle video files - Gemini can take videos directly
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_data = encode_image(path)  # Read the entire video file
                # Map file extension to MIME type
                ext = path.split('.')[-1].lower()
                mime_map = {
                    'mp4': 'video/mp4',
                    'avi': 'video/x-msvideo',
                    'mov': 'video/quicktime',
                    'mkv': 'video/x-matroska',
                    'mpeg': 'video/mpeg',
                    'mpg': 'video/mpeg',
                    'wmv': 'video/x-ms-wmv',
                    'webm': 'video/webm',
                    '3gpp': 'video/3gpp'
                }
                mime_type = mime_map.get(ext, 'video/mp4')
                
                loaded_data.append({
                    'path': path,
                    'type': 'video',
                    'data': video_data,
                    'mime_type': mime_type
                })
            # Handle list of frames
            elif isinstance(path, list):
                loaded_data.append({
                    'path': path,
                    'type': 'frame_list',
                    'frames': []
                })
            # Handle image files
            else:
                mime_type = get_image_type(path)
                loaded_data.append({
                    'path': path,
                    'type': 'image',
                    'data': encode_image(path),
                    'mime_type': mime_type
                })
        return loaded_data

    def forward_single(self, data, question, answer):
        try:
            config = GenerateContentConfig(
                temperature=0,
                top_p=0.95,
                top_k=20,
                response_logprobs=True,
                logprobs=3 #self.logprobs
            )
            
            parts = []
            
            # Add the question
            parts.append(question)
            
            # Add the visual content
            if data['type'] == 'video':
                # For video, add the entire video as one part
                parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
            elif data['type'] == 'frame_list' and 'frames' in data:
                # For frame lists, add each frame
                for frame in data['frames']:
                    parts.append(Part.from_bytes(data=frame, mime_type="image/jpeg"))
            else:
                # For images
                parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=parts,
                config=config,
            )

            # Get the set of chosen candidates' logprobs:
            candidates = response.candidates[0].logprobs_result.top_candidates[0].candidates

            ans_prob = 0.0
            target_answer = answer.lower().strip()
            for candidate in candidates:
                cur_token = candidate.token.lower().strip()

                if target_answer in cur_token:
                    ans_prob = max(ans_prob, np.exp(candidate.log_probability))

            return torch.tensor([ans_prob])
                
        except Exception as e:
            try:  # Second try
                config = GenerateContentConfig(
                    temperature=0,
                    top_p=0.95,
                    top_k=20,
                    response_logprobs=True,
                    logprobs=3 #self.logprobs
                )
                
                parts = []
                
                # Add the question
                parts.append(question)
                
                # Add the visual content
                if data['type'] == 'video':
                    # For video, add the entire video as one part
                    parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
                elif data['type'] == 'frame_list' and 'frames' in data:
                    # For frame lists, add each frame
                    for frame in data['frames']:
                        parts.append(Part.from_bytes(data=frame, mime_type="image/jpeg"))
                else:
                    # For images
                    parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    config=config,
                )

                # Get the set of chosen candidates' logprobs:
                candidates = response.candidates[0].logprobs_result.top_candidates[0].candidates

                ans_prob = 0.0
                target_answer = answer.lower().strip()
                for candidate in candidates:
                    cur_token = candidate.token.lower().strip()

                    if target_answer in cur_token:
                        ans_prob = max(ans_prob, np.exp(candidate.log_probability))
                print(f'Answer Probability {ans}')
                return torch.tensor([ans_prob])
                        
            except Exception as retry_error:
                print(f"Failed media: {data['path']} and question: {question} and answer: {answer}")
                print(f"Error: {str(e)}")
                print(f"Retry error: {str(retry_error)}")
                return torch.tensor([0.0])

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = default_question_template,
                answer_template: str = default_answer_template,
                num_frames: int = 4) -> torch.Tensor:

        assert len(paths) == len(texts), "Number of paths and texts must match"

        self.load_model()

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        loaded_data = self.load_images(paths, num_frames)

        lm_prob = torch.zeros(len(paths))

        for idx, (data, question, answer) in enumerate(zip(loaded_data, questions, answers)):
            lm_prob[idx] = self.forward_single(data, question, answer)

        return lm_prob
    
    def generate_single(self, data, question, max_new_tokens=1024):
        try:
            config = GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=max_new_tokens,
            )
            
            parts = []
            
            # Add the question
            parts.append(question)
            
            # Add the visual content
            if data['type'] == 'video':
                # Direct video input
                parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
            elif data['type'] == 'frame_list' and 'frames' in data:
                # Multiple frames
                for frame in data['frames']:
                    parts.append(Part.from_bytes(data=frame, mime_type="image/jpeg"))
            else:
                # Single image
                parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=parts,
                # config=config,
            )
            
            return response.text

        except Exception as e:
            try:  # Second try
                config = GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=max_new_tokens,
                )
                
                parts = []
                
                # Add the question
                parts.append(question)
                
                # Add the visual content - same pattern as above for retry
                if data['type'] == 'video':
                    parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
                elif data['type'] == 'frame_list' and 'frames' in data:
                    for frame in data['frames']:
                        parts.append(Part.from_bytes(data=frame, mime_type="image/jpeg"))
                else:
                    parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    # config=config,
                )
                
                return response.text
                
            except Exception as retry_error:
                print(f"Failed media: {data['path']} and question: {question}")
                print(f"Error: {str(e)}")
                print(f"Retry error: {str(retry_error)}")
                return ""

    def generate(self,
            images: List[str],
            texts: List[str],
            num_frames: int = 5,
            max_new_tokens: int = 1024) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        # self.load_model()
        
        questions = texts
        loaded_data = self.load_images(images, num_frames)

        generated_outputs = []

        for idx, (data, question) in enumerate(zip(loaded_data, questions)):
            generated_text = self.generate_single(data, question, max_new_tokens)
            generated_outputs.append(generated_text)

        return generated_outputs