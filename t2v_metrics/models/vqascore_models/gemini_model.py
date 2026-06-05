import math
import cv2
import numpy as np
from typing import List
import torch
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    ThinkingConfig,
)

from .vqa_model import VQAScoreModel

import os

default_question_template = 'Does this figure show "{}"? Please answer Yes or No.',
default_answer_template = 'Yes'

GEMINI_MODELS = {
    # Still active — older stable (working VQAScore)
    'gemini-2.5-flash':          {},
    'gemini-2.5-pro':            {},

    # Current generation (logprobs/VQAScore not supported)
    # 'gemini-3.1-pro-preview':    {},  # Feb 2026, current flagship
    # 'gemini-3-flash-preview': {},  #
    # 'gemini-3.5-flash': {}

}

SAFETY_SETTINGS = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,          threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,   threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,   threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
]


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


def get_image_type(image_path):
    image_type = image_path.split('.')[-1].lower()
    assert image_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']
    return f"image/{image_type}"


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


def find_first_output_token_index(chosen_candidates) -> int | None:
    """
    Gemini thinking models wrap internal reasoning in a block:
        <ctrl94> thought ... </thought> <ctrl95>
    Toggles in_thinking on each <ctrl token and skips everything inside.
    Returns the index of the first real output token, or None if not found.
    """
    in_thinking = False
    for i, chosen in enumerate(chosen_candidates):
        if chosen.token.startswith('<ctrl'):
            in_thinking = not in_thinking
            continue
        if in_thinking:
            continue
        return i
    return None


class GeminiModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True

    def __init__(self,
             model_name='gemini-2.5-pro',
             device='cuda',
             cache_dir="./cache_dir",
             project_id=None,
             api_key=None,
             location=None,
             logprobs=5):
        assert model_name in GEMINI_MODELS, \
            f"Model {model_name} not supported. Choose from {list(GEMINI_MODELS.keys())}"

        # Resolve credentials — explicit args take priority over env vars
        project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        api_key    = api_key    or os.environ.get("GEMINI_API_KEY")
        location   = location  or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

        assert project_id is not None or api_key is not None, (
            "No Gemini credentials provided. Either:\n"
            "  - Pass project_id= or set GOOGLE_CLOUD_PROJECT (Vertex AI, uses ADC auth)\n"
            "  - Pass api_key= or set GEMINI_API_KEY (Gemini Developer API)"
        )

        # If both are provided, Vertex AI takes priority
        self.project_id = project_id
        self.api_key    = api_key if project_id is None else None
        self.location   = location
        self.logprobs   = logprobs
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        if self.project_id is not None:
            # Vertex AI — authentication via ADC (gcloud auth application-default login)
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
        else:
            # Gemini Developer API — authentication via API key from Google AI Studio
            self.client = genai.Client(api_key=self.api_key)

    def load_images(self, paths: List[str], num_frames: int = None) -> List[dict]:
        loaded_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
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
                    '3gpp': 'video/3gpp',
                }
                loaded_data.append({
                    'path': path,
                    'type': 'video',
                    'data': encode_image(path),
                    'mime_type': mime_map.get(ext, 'video/mp4'),
                })
            elif isinstance(path, list):
                loaded_data.append({
                    'path': path,
                    'type': 'frame_list',
                    'frames': [],
                })
            else:
                loaded_data.append({
                    'path': path,
                    'type': 'image',
                    'data': encode_image(path),
                    'mime_type': get_image_type(path),
                })
        return loaded_data

    def _build_parts(self, data, question):
        parts = [question]
        if data['type'] == 'video':
            parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
        elif data['type'] == 'frame_list' and 'frames' in data:
            for frame in data['frames']:
                parts.append(Part.from_bytes(data=frame, mime_type="image/jpeg"))
        else:
            parts.append(Part.from_bytes(data=data['data'], mime_type=data['mime_type']))
        return parts

    def forward_single(self, data, question, answer, temperature=0.0):
        config = GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            response_logprobs=True,
            logprobs=self.logprobs,
            max_output_tokens=65536,   # headroom for thinking models
            safety_settings=SAFETY_SETTINGS,
        )

        for attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self._build_parts(data, question),
                    config=config,
                )

                logprobs_result = response.candidates[0].logprobs_result
                if logprobs_result is None:
                    print(f"logprobs_result is None for {data['path']}")
                    return torch.tensor([0.0])

                # Find first real output token (skip thinking tokens)
                chosen = logprobs_result.chosen_candidates
                top = logprobs_result.top_candidates
                first_idx = find_first_output_token_index(chosen)

                if first_idx is None:
                    print(f"No output token found for {data['path']} — try increasing max_output_tokens")
                    return torch.tensor([0.0])

                target = answer.lower().strip()
                ans_prob = 0.0
                for candidate in top[first_idx].candidates:
                    cur_token = candidate.token.lower().strip()
                    if target in cur_token:
                        ans_prob = max(ans_prob, math.exp(candidate.log_probability))

                return torch.tensor([ans_prob])

            except Exception as e:
                if attempt == 0:
                    print(f"Attempt 1 failed for {data['path']}: {e}. Retrying...")
                else:
                    print(f"Both attempts failed for {data['path']}: {e}")
                    return torch.tensor([0.0])

    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str = default_question_template,
                answer_template: str = default_answer_template,
                num_frames: int = 4,
                fps=None,
                temperature: float = 1.0) -> torch.Tensor:

        if self.project_id is None:
            raise ValueError(
                "Gemini VQAScore requires Vertex AI (logprobs not supported via "
                "the Gemini Developer API). Provide a project_id or set GOOGLE_CLOUD_PROJECT."
            )

        assert len(images) == len(texts), "Number of images and texts must match"

        self.load_model()

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        loaded_data = self.load_images(images, num_frames)

        lm_prob = torch.zeros(len(images))
        for idx, (data, question, answer) in enumerate(zip(loaded_data, questions, answers)):
            lm_prob[idx] = self.forward_single(data, question, answer, temperature=temperature)

        return lm_prob

    def generate_single(self, data, question, max_new_tokens=65536, temperature=0.0):
        config = GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=max_new_tokens,
            safety_settings=SAFETY_SETTINGS,
        )

        for attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self._build_parts(data, question),
                    config=config,
                )
                return response.text

            except Exception as e:
                if attempt == 0:
                    print(f"Attempt 1 failed for {data['path']}: {e}. Retrying...")
                else:
                    print(f"Both attempts failed for {data['path']}: {e}")
                    return ""

    def generate(self,
                 images: List[str],
                 texts: List[str],
                 num_frames: int = 5,
                 max_new_tokens: int = 65536,
                 fps=None,
                 temperature: float = 0.0) -> List[str]:

        assert len(images) == len(texts), "Number of paths and texts must match"

        self.load_model()

        loaded_data = self.load_images(images, num_frames)

        return [
            self.generate_single(data, question, max_new_tokens=max_new_tokens, temperature=temperature)
            for data, question in zip(loaded_data, texts)
        ]