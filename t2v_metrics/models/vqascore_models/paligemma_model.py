import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from decord import VideoReader, cpu

from .vqa_model import VQAScoreModel

PALIGEMMA_MODELS = {
    'paligemma-3b-mix-224': {
        'processor': {
            'path': 'google/paligemma-3b-mix-224',
        },
        'model': {
            'path': 'google/paligemma-3b-mix-224',
        },
    },
    'paligemma-3b-mix-448': {
        'processor': {
            'path': 'google/paligemma-3b-mix-448',
        },
        'model': {
            'path': 'google/paligemma-3b-mix-448',
        },
    },
    'paligemma-3b-mix-896': {
        'processor': {
            'path': 'google/paligemma-3b-mix-896',
        },
        'model': {
            'path': 'google/paligemma-3b-mix-896',
        },
    }
}

class PaliGemmaModel(VQAScoreModel):
    video_mode = "concat" # Paligemma's pytorch port does not have video inference natively
    allows_image = True
    def __init__(self,
                 model_name='paligemma-3b-mix-448',
                 device='cuda',
                 cache_dir=None):
        assert model_name in PALIGEMMA_MODELS, f"Model {model_name} not found in PALIGEMMA_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = PALIGEMMA_MODELS[model_name]
        self.load_model()

    def load_model(self):
        model_path = self.model_info['model']['path']
        processor_path = self.model_info['processor']['path']
        
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(processor_path)

    def load_images(self, paths: List[str], num_frames: int = 16) -> List[Union[Image.Image, torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                video_frames = self.load_video(path, num_frames)
                processed_data.append(video_frames)
            elif path.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    processed_data.append(image)
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [torch.from_numpy(frame).permute(2, 0, 1) for frame in np_array]
                    processed_data.append(frames)
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                processed_data.append(image)
        return processed_data

    def load_video(self, video_path, num_frames):
        raise NotImplementedError("Direct video processing is not supported for PaliGemma's PyTorch port.")

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = "Does this figure show \"{}\"? Please answer yes or no.",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(paths)

        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
            model_inputs = self.processor(text=question, images=data, return_tensors="pt")
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.processor.tokenizer.encode(answer)[0]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)

        return torch.tensor(lm_probs)
    
    def generate(self,
            images: List[str],
            texts: List[str],
            max_new_tokens: int = 256) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"

        processed_data = self.load_images(images)

        generated_texts = []
        for data, text in zip(processed_data, texts):
            model_inputs = self.processor(text=text, images=data, return_tensors="pt")
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens
                )
                
                text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(text.strip())

        return generated_texts
