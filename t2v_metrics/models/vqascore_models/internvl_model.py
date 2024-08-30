import torch
import numpy as np
from PIL import Image
from typing import List, Union
import copy
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer

from .vqa_model import VQAScoreModel

INTERNVL2_MODELS = {
    'internvl2-8b': {
        'path': 'OpenGVLab/InternVL2-8B',
        'dtype': torch.bfloat16,
        'use_flash_attn': True,
    },
    # Add other InternVL2 models here if needed
}

class InternVL2Model(VQAScoreModel):
    def __init__(self,
                 model_name='internvl2-8b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in INTERNVL2_MODELS, f"Model {model_name} not found in INTERNVL2_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = INTERNVL2_MODELS[model_name]
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        model_path = self.model_info['path']
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.model_info['dtype'],
            low_cpu_mem_usage=True,
            use_flash_attn=self.model_info['use_flash_attn'],
            trust_remote_code=True
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

    def load_images(self, paths: List[str], max_num: int = 12, num_segments: int = 8) -> List[torch.Tensor]:
        processed_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                pixel_values, num_patches_list = self.load_video(path, num_segments=num_segments, max_num=max_num)
                processed_data.append((pixel_values, num_patches_list))
            else:  # Image file
                pixel_values = self.load_image(path, max_num=max_num)
                processed_data.append((pixel_values, [pixel_values.shape[0]]))
        return processed_data

    def load_image(self, image_path, max_num=12):
        # Implement image loading and preprocessing here
        # This is a placeholder and should be replaced with actual implementation
        return torch.randn(max_num, 3, 224, 224).to(self.model_info['dtype']).to(self.device)

    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=8):
        # Implement video loading and preprocessing here
        # This is a placeholder and should be replaced with actual implementation
        pixel_values = torch.randn(num_segments * max_num, 3, input_size, input_size).to(self.model_info['dtype']).to(self.device)
        num_patches_list = [max_num] * num_segments
        return pixel_values, num_patches_list

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = "Does this figure show \"{}\"? Please answer yes or no.",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        processed_data = self.load_images(paths)
        
        lm_probs = []
        for (pixel_values, num_patches_list), question in zip(processed_data, questions):
            if len(num_patches_list) > 1:  # Video
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                question = video_prefix + question
            else:  # Image
                question = '<image>\n' + question

            input_ids, attention_mask = self.prepare_inputs(question, num_patches_list)
            
            outputs = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.tokenizer.encode("Yes")[0]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)
        
        return torch.tensor(lm_probs)

    def prepare_inputs(self, question, num_patches_list):
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            question = question.replace('<image>', image_tokens, 1)
        
        model_inputs = self.tokenizer(question, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        
        return input_ids, attention_mask