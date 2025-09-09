import os 
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Union
import copy
import warnings
import requests
import tempfile

from .tarsier.tasks.utils import load_model_and_processor
from .tarsier.dataset.custom_data_parsers.utils import put_pred_to_data_dict, get_prompt_from_data_dict
from .tarsier.dataset.utils import format_one_sample, get_visual_type
# sys.path = sys.path[2:]
# print(sys.path)
import yaml


from .vqa_model import VQAScoreModel

warnings.filterwarnings("ignore")

current_path = os.path.dirname(os.path.abspath(__file__))
TARSIER_MODELS = {
    'tarsier-recap-7b': {
        'model': {
            'path': "omni-research/Tarsier2-Recap-7b" ,
            'config': f'{current_path}/tarsier/configs/tarser2_default_config.yaml',
        },
    },

    'tarsier2-7b': {
        'model': {
            'path': "omni-research/Tarsier2-7b-0115" ,
            'config': f'{current_path}/tarsier/configs/tarser2_default_config.yaml',
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

        self.tmp_files = []

    def load_model(self):
        # sys.path = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vqascore_models/tarsier')] + sys.path
        # sys.path = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vqascore_models/tarsier/models')] + sys.path
        # print(f'sys.path {sys.path}')
        model_path = self.model_info['model']['path']
        config_path = self.model_info['model']['config']
        
        data_config = yaml.safe_load(open(config_path, 'r'))
        self.model, self.processor = load_model_and_processor(model_path, data_config=data_config)
        self.model.eval()
        # sys.path = sys.path[2:]

    def load_images(self, prompt, video_file, generate_kwargs):
        if video_file.startswith(("http://", "https://")):
            tmpfile = None
            try:
                # 1. Read video
                response = requests.get(video_file, stream=True)
                response.raise_for_status()

                # 2. Create tempfile to write  video to
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    self.tmp_files.append(tmpfile.name)

                    for chunk in response.iter_content(chunk_size=8192):
                        tmpfile.write(chunk)

                    
                    tmpfile.flush()
                # 3. Replace video file link with the temp file path name
                    
                    video_file=tmpfile.name
            except Exception as e:
                if tmpfile:
                    os.remove(tmpfile.name)
                    self.tmp_files.remove(tmpfile.name)

                
                print(f"Exception while downloading video {video_file} \n\nError: {e}")
                raise

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
    def clean_temp_files(self):
        # Clean up tempfiles created:
        for file_path in self.tmp_files:
            try:
                os.remove(file_path)
            except OSError:
                pass
        self.tmp_files.clear()
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
            answer = answer_template.format(text)
            
            generate_kwargs = {
                "do_sample": False,
                "max_new_tokens": 1,
                "top_p": 1.0,
                "temperature": 0,
                "use_cache": True,
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            
            _, model_inputs = self.load_images(question, path, generate_kwargs)
            
            outputs = self.model.generate(
                **model_inputs,
                **generate_kwargs,
            )
            
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.processor.processor.tokenizer.encode(answer)[0]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)

        self.clean_temp_files()

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
            
            output_text, _ = self.load_images(text, path, generate_kwargs)
            generated_outputs.append(output_text.strip())

        
        self.clean_temp_files()

        return generated_outputs