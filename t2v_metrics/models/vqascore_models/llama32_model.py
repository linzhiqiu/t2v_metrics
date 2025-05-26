import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import List, Union
import numpy as np

LLAMA_32_VISION_MODELS = {
    'llama-3.2-1b': {
        'model': {
            'path': 'meta-llama/Llama-3.2-1B',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-3.2-1B'},
    },
    'llama-3.2-3b': {
        'model': {
            'path': 'meta-llama/Llama-3.2-3B',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-3.2-3B'},
    },
    'llama-3.2-1b-instruct': {
        'model': {
            'path': 'meta-llama/Llama-3.2-1B-Instruct',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-3.2-1B-Instruct'},
    },
    'llama-3.2-3b-instruct': {
        'model': {
            'path': 'meta-llama/Llama-3.2-3B-Instruct',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-3.2-3B-Instruct'},
    },
    'llama-guard-3-1b': {
        'model': {
            'path': 'meta-llama/Llama-Guard-3-1B',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-Guard-3-1B'},
    },
    'llama-3.2-11b-vision': {
        'model': {
            'path': 'meta-llama/Llama-3.2-11B-Vision',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-3.2-11B-Vision'},
    },
    'llama-3.2-11b-vision-instruct': {
        'model': {
            'path': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-3.2-11B-Vision-Instruct'},
    },
    'llama-3.2-90b-vision': {
        'model': {
            'path': 'meta-llama/Llama-3.2-90B-Vision',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-3.2-90B-Vision'},
    },
    'llama-3.2-90b-vision-instruct': {
        'model': {
            'path': 'meta-llama/Llama-3.2-90B-Vision-Instruct',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-3.2-90B-Vision-Instruct'},
    },
    'llama-guard-3-11b-vision': {
        'model': {
            'path': 'meta-llama/Llama-Guard-3-11B-Vision',
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
        },
        'processor': {'path': 'meta-llama/Llama-Guard-3-11B-Vision'},
    },
}

class LLaMA32VisionModel:
    video_mode = "concat"
    allows_image = True
    def __init__(self,
                 model_name='llama-3.2-11b-vision-instruct',
                 device='cuda',
                 cache_dir=None):
        assert model_name in LLAMA_32_VISION_MODELS, f"Model {model_name} not found in LLAMA_32_VISION_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = LLAMA_32_VISION_MODELS[model_name]
        self.load_model()
        

    def load_model(self):
        model_config = self.model_info['model']
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_config['path'],
            torch_dtype=model_config['torch_dtype'],
            device_map=model_config['device_map'],
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_info['processor']['path'],
            cache_dir=self.cache_dir
        )

    def load_images(self, paths: List[str]) -> List[Image.Image]:
        processed_data = []
        for path in paths:
            if path.startswith(("http://", "https://")):
                raise NotImplementedError("Web link image/video inputs are not yet supported for this model. Please use a local path, or otherwise, make a Github issue request if this feature is necessary.")
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                raise NotImplementedError("Video processing is not supported for LLaMA-3.2 Vision model.")
            elif path.lower().endswith('.npy'):
                raise NotImplementedError("NumPy array processing is not implemented for LLaMA-3.2 Vision model.")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                processed_data.append(image)
        return processed_data

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = "Does this image show \"{}\"? Answer the question with Yes or No",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        images = self.load_images(paths)
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        lm_probs = []
        for image, question, answer in zip(images, questions, answers):
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.processor.tokenizer.encode(answer)[1]
    
            lm_probs.append(lm_prob)
        
        return torch.tensor(lm_probs)

    def generate(self,
            images: List[str],
            texts: List[str],
            max_new_tokens: int = 256) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        images = self.load_images(images)
        questions = texts
        
        generated_outputs = []
        for image, question in zip(images, questions):
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens
                )
                
                text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_outputs.append(text.strip())
        
        return generated_outputs
    def load_video(self, video_path, max_frames_num):
        raise NotImplementedError("Video processing is not supported for LLaMA-3.2 Vision model.")
