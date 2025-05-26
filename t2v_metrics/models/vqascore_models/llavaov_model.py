import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Union
import copy
from decord import VideoReader, cpu

from .vqa_model import VQAScoreModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates

LLAVA_OV_MODELS = {
    'llava-onevision-qwen2-7b-si': {
        'tokenizer': {
            'path': 'lmms-lab/llava-onevision-qwen2-7b-si',
        },
        'model': {
            'path': 'lmms-lab/llava-onevision-qwen2-7b-si',
            'conversation': 'qwen_1_5',
            'image_aspect_ratio': 'pad',
        },
    },
    'llava-onevision-qwen2-7b-ov': {
        'tokenizer': {
            'path': 'lmms-lab/llava-onevision-qwen2-7b-ov',
        },
        'model': {
            'path': 'lmms-lab/llava-onevision-qwen2-7b-ov',
            'conversation': 'qwen_1_5',
            'image_aspect_ratio': 'pad',
        },
    },
}

class LLaVAOneVisionModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='llava-onevision-qwen2-7b-ov',
                 device='cuda',
                 cache_dir=None):
        assert model_name in LLAVA_OV_MODELS, f"Model {model_name} not found in LLAVA_OV_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = LLAVA_OV_MODELS[model_name]
        self.conversational_style = self.model_info['model']['conversation']
        self.load_model()
        

    def load_model(self):
        model_path = self.model_info['model']['path']
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto", attn_implementation="sdpa")
        self.model.eval()

    def load_images(self, paths: List[str], num_frames: int = 16) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        for path in paths:
            if path.startswith(("http://", "https://")):
                raise NotImplementedError("Web link image/video inputs are not yet supported for this model. Please use a local path, or otherwise, make a Github issue request if this feature is necessary.")
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                video_frames = self.load_video(path, num_frames)
                frames = self.processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(self.device)
                processed_data.append(frames)
            elif path.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    image_tensor = process_images([image], self.processor, self.model.config)
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    processed_data.append(image_tensor[0])
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [Image.fromarray(frame.astype('uint8'), 'RGB') for frame in np_array]
                    frames_tensor = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device)
                    processed_data.append(frames_tensor)
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                image_tensor = process_images([image], self.processor, self.model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                processed_data.append(image_tensor[0])
        return processed_data

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).numpy()
        return spare_frames

    def forward(self,
                paths: List[str],
                texts: List[str],
                num_frames: int=16,
                question_template: str = "Does this image show \"{}\"? Answer the question with Yes or No",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        questions = [self.format_question(question) for question in questions]
       
        processed_data = self.load_images(paths, num_frames)
        
        prompts = [qs for qs in questions]

        lm_probs = []
        for data, prompt, answer in zip(processed_data, prompts, answers):
            if isinstance(data, torch.Tensor) and data.dim() == 4:  # Video
                image_sizes = [data.shape[2:] for _ in range(data.shape[0])]
                modalities = ["video"]
            else:  # Image
                image_sizes = [data.shape[1:]]
                modalities = None
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                images=[data],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=1,
                modalities=modalities,
                output_scores=True,
                return_dict_in_generate=True,
            )
            
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.tokenizer.encode(answer)[0]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)

        
        return torch.tensor(lm_probs)
    
    def generate(self,
            images: List[str],
            texts: List[str],
            num_frames: int = 12,
            max_new_tokens: int = 256,) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        texts = [self.format_question(text) for text in texts]
        processed_data = self.load_images(images, num_frames)
        generated_texts = []
        for data, prompt in zip(processed_data, texts):
            if isinstance(data, torch.Tensor) and data.dim() == 4:  # Video
                image_sizes = [data.shape[2:] for _ in range(data.shape[0])]
                modalities = ["video"]
            else:  # Image
                image_sizes = [data.shape[1:]]
                modalities = None
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                images=[data],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                modalities=modalities
            )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(text.strip())

        return generated_texts

    def format_question(self, question):
        conv = copy.deepcopy(conv_templates[self.conversational_style])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

