import torch
import numpy as np
from PIL import Image
from typing import List, Union
import copy
import warnings
from tarsier.tasks.utils import load_model_and_processor
from tarsier.dataset.custom_data_parsers.utils import put_pred_to_data_dict, get_prompt_from_data_dict
from tarsier.dataset.utils import format_one_sample, get_visual_type
import yaml

from .vqa_model import VQAScoreModel

warnings.filterwarnings("ignore")

TARSIER_MODELS = {
    'tarsier-recap-7b': {
        'model': {
            'path': 'path_to_tarsier_model',  # Replace with actual path
            'config': 'configs/tarser2_default_config.yaml',
        },
    },
}

class TarsierModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='tarsier-recap-7b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in TARSIER_MODELS, f"Model {model_name} not found in TARSIER_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = TARSIER_MODELS[model_name]
        self.load_model()

    def load_model(self):
        model_path = self.model_info['model']['path']
        config_path = self.model_info['model']['config']
        
        data_config = yaml.safe_load(open(config_path, 'r'))
        self.model, self.processor = load_model_and_processor(model_path, data_config=data_config)
        self.model.eval()

    def process_one(self, prompt, video_file, generate_kwargs):
        sample = format_one_sample(video_file, prompt)
        batch_data = self.processor(sample)
        
        model_inputs = {}
        for k, v in batch_data.items():
            if not isinstance(v, torch.Tensor):
                continue
            model_inputs[k] = v.to(self.model.device)
            
        outputs = self.model.generate(
            **model_inputs,
            **generate_kwargs,
        )
        
        output_text = self.processor.processor.tokenizer.decode(
            outputs[0][model_inputs['input_ids'][0].shape[0]:], 
            skip_special_tokens=True
        )
        return output_text, model_inputs

    def forward(self,
                images: List[str],
                texts: List[str],
                num_frames: int=16,
                question_template: str = "Does this video show \"{}\"? Answer the question with Yes or No.",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(images) == len(texts), "Number of images and texts must match"
    
        lm_probs = []
        for path, text in zip(images, texts):
            if not get_visual_type(path) in ['video', 'gif']:
                print(f'This model supports only video inference. {path} is invalid')
                lm_probs.append(0.0)
                continue

            question = question_template.format(text)
            
            generate_kwargs = {
                "do_sample": False,
                "max_new_tokens": 1,
                "top_p": 1.0,
                "temperature": 0,
                "use_cache": True,
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            
            _, model_inputs = self.process_one(question, path, generate_kwargs)
            
            outputs = self.model.generate(
                **model_inputs,
                **generate_kwargs,
            )
            
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.processor.processor.tokenizer.encode("Yes")[0]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)

        return torch.tensor(lm_probs)
    
    def generate(self,
                images: List[str],
                texts: List[str],
                num_frames: int = 4,
                max_new_tokens: int = 256) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        generated_outputs = []
        for path, text in zip(images, texts):
            if not get_visual_type(path) in ['video', 'gif']:
                print(f'This model supports only video inference. {path} is invalid')
                generated_outputs.append("")
                continue
                
            generate_kwargs = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "top_p": 1.0,
                "temperature": 0,
                "use_cache": True
            }
            
            output_text, _ = self.process_one(text, path, generate_kwargs)
            generated_outputs.append(output_text.strip())

        return generated_outputs