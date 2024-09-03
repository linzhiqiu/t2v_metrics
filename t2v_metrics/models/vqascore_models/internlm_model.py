import torch
import numpy as np
import os
import tempfile
from typing import List, Union
from transformers import AutoModel, AutoTokenizer

from .vqa_model import VQAScoreModel

INTERNLMXCOMPOSER25_MODELS = {
    'internlmxcomposer25-7b': {
        'tokenizer': {
            'path': 'internlm/internlm-xcomposer2d5-7b',
            'trust_remote_code': True,
        },
        'model': {
            'pretrained_model_name_or_path': 'internlm/internlm-xcomposer2d5-7b',
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
        },
    },

}

class InternLMXComposer25Model(VQAScoreModel):
    def __init__(self,
                 model_name='internlmxcomposer25-7b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in INTERNLMXCOMPOSER25_MODELS, f"Model {model_name} not found in INTERNLMXCOMPOSER25_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = INTERNLMXCOMPOSER25_MODELS[model_name]
        self.load_model()

    def load_model(self):
        tokenizer_path = self.model_info['tokenizer']['path']
        
        self.model = AutoModel.from_pretrained(
            **self.model_info['model']
        ).eval().to(self.device).half()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **self.model_info['tokenizer']
        )
        self.model.tokenizer = self.tokenizer

    def process_path(self, path: str) -> str:
        if path.lower().endswith('.npy'):
            # Create a temporary file with .jpg extension
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
            
            # Load the .npy file and save it as a temporary .jpg file
            np_array = np.load(path)
            if np_array.ndim == 3 and np_array.shape[2] == 3:  # Ensure it's an RGB image
                from PIL import Image
                Image.fromarray(np_array.astype('uint8'), 'RGB').save(temp_path)
            else:
                raise ValueError(f"Unexpected shape for NumPy array in {path}")
            
            return temp_path
        else:
            return path

    def load_images(self, paths: List[str]) -> List[str]:
        processed_paths = []
        for path in paths:
            processed_paths.append(self.process_path(path))
        return processed_paths

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = "Does this figure show \"{}\"? Please answer yes or no.",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        processed_paths = self.load_images(paths)

        lm_probs = []
        temp_files = []
        for path, question in zip(processed_paths, questions):
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                query = f"Here are some frames of a video. {question}"
            else:  # Image file
                query = f"<ImageHere> {question}"

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                response, _ = self.model.chat(self.tokenizer, query, [path], do_sample=False, num_beams=3, use_meta=True)

            # Since we don't have direct access to logits, we'll use a simple heuristic
            # to estimate the probability of a "Yes" answer
            lm_prob = 1.0 if response.lower().startswith("yes") else 0.0
            lm_probs.append(lm_prob)

            # If this is a temporary file, add it to the list for cleanup
            if path.startswith(tempfile.gettempdir()):
                temp_files.append(path)

        # Clean up temporary files
        for temp_file in temp_files:
            os.remove(temp_file)

        return torch.tensor(lm_probs)