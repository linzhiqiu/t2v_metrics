import torch
import numpy as np
from PIL import Image
from typing import List, Union
import copy
import time
from decord import VideoReader, cpu
import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = os.path.join(current_dir, "perceptionlm")
# print(f'current_dir {current_dir}')
import sys
import tempfile
import requests

# sys.path.append(current_dir)
from .perceptionlm.core.args import dataclass_from_dict
from .perceptionlm.core.transforms.image_transform import get_image_transform
from .perceptionlm.core.transforms.video_transform import get_video_transform
from .perceptionlm.apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer


PERCEPTION_LM_MODELS = {
    'perception-lm-1b': {
        'path': 'facebook/Perception-LM-1B',
        'image_size': 448,
        'max_video_frames': 32,
        'max_num_tiles': 36,
    },
    'perception-lm-3b': {
        'path': 'facebook/Perception-LM-3B',
        'image_size': 448,
        'max_video_frames': 32,
        'max_num_tiles': 36,
    },
    'perception-lm-8b': {
        'path': 'facebook/Perception-LM-8B',
        'image_size': 448,
        'max_video_frames': 32,
        'max_num_tiles': 36,
    },
}


class PerceptionLMModel:
    video_mode = "direct"
    allows_image = True
    
    def __init__(self,
                 model_name='perception-lm-8b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in PERCEPTION_LM_MODELS, f"Model {model_name} not found in PERCEPTION_LM_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = PERCEPTION_LM_MODELS[model_name]
        self.load_model()

        self.tmp_files = []
        
    def load_model(self):
        model_path = self.model_info['path']
        self.model, self.tokenizer, self.config = load_consolidated_model_and_tokenizer(model_path)
        self.model.eval()

    def load_images(self, paths: List[str], num_tiles: int = 4) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                video_frames = self.load_video(path, num_tiles)
                processed_data.append(video_frames)
            elif path.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    transform = get_image_transform(
                        vision_input_type=("vanilla" if num_tiles == 1 else self.config.data.vision_input_type),
                        image_res=self.model.vision_model.image_size,
                        max_num_tiles=num_tiles,
                    )
                    image_tensor, _ = transform(image)
                    processed_data.append(image_tensor)
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [Image.fromarray(frame.astype('uint8'), 'RGB') for frame in np_array]
                    transform = get_video_transform(
                        image_res=self.model.vision_model.image_size,
                    )
                    frames_tensor, _ = transform((frames, len(frames), None, None, None))
                    processed_data.append(frames_tensor)
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                transform = get_image_transform(
                    vision_input_type=("vanilla" if num_tiles == 1 else self.config.data.vision_input_type),
                    image_res=self.model.vision_model.image_size,
                    max_num_tiles=num_tiles,
                )
                image_tensor, _ = transform(image)
                processed_data.append(image_tensor)
        return processed_data

    def load_video(self, video_path, num_frames):
        if video_path.startswith(("http://", "https://")):
            tmpfile = None  

            try:

                response = requests.get(video_path, stream=True)
                response.raise_for_status()


                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:

                    self.tmp_files.append(tmpfile.name)
                    for chunk in response.iter_content(chunk_size=8192):
                        tmpfile.write(chunk)
                    
                    tmpfile.flush()
                    video_path = tmpfile.name
            
            except Exception as e:
                if tmpfile:
                    os.remove(tmpfile.name)
                    self.tmp_files.remove(tmpfile.name)
                
                
                print(f'Error when downloading video {video_path} \n\nError: {e}')

                raise
                
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, num_frames, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        video_frames = vr.get_batch(frame_idx).numpy()
        
        # Convert to PIL Image list
        frames = [Image.fromarray(frame) for frame in video_frames]
        # Apply video transform
        transform = get_video_transform(
            image_res=self.model.vision_model.image_size,
        )
        video_tensor, _ = transform((video_path, num_frames, None, None, None))
        return video_tensor

    def clear_temp_files(self):
        for file_name in self.tmp_files:
            try:
                os.remove(file_name)
            except OSError:
                pass
        self.tmp_files.clear()
    
    def forward(self,
                paths: List[str],
                texts: List[str],
                num_frames: int = 32,
                num_tiles: int = 36,
                question_template: str = "Does this image show \"{}\"? Answer the question with Yes or No",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template]
        
        processed_data = self.load_images(paths, num_tiles)
        
        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
            # Determine if this is a video or image
            media_type = "video" if isinstance(data, torch.Tensor) and data.dim() > 3 else "image"
            
            # Create prompt
            prompts = [(question, data)]
            
            # Create generator
            gen_cfg = dataclass_from_dict(
                PackedCausalTransformerGeneratorArgs,
                {"temperature": 0.0, "top_p": None, "top_k": None},
                strict=False,
            )
            generator = PackedCausalTransformerGenerator(gen_cfg, self.model, self.tokenizer)
            
            # Run generation for scoring (force output to match expected answer)
            _, _, _, logits = generator.generate(prompts)#, force_output=answer)
            print(f'Vocabulary Length {len(logits[0].shape)}')

            ids, _ = self.tokenizer._tokenize_for_generation(question="Yes", media=data)
            yes_id = ids[0]
            print(f'Yes Token ID {yes_id}')

            # Use loglikelihood as probability score
            lm_prob = torch.exp(logits[yes_id]).item()
            lm_probs.append(lm_prob)
        
        self.clear_temp_files()
        return torch.tensor(lm_probs)
    
    def generate(self,
            images: List[str],
            texts: List[str],
            num_frames: int = 32,
            num_tiles: int = 36,
            temperature: float = 0.0,
            top_p: float = None,
            top_k: int = None,
            max_new_tokens: int = 256) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        processed_data = self.load_images(images, num_tiles)
        generated_texts = []
        
        for data, prompt in zip(processed_data, texts):
            # Determine if this is a video or image
            media_type = "video" if isinstance(data, torch.Tensor) and data.dim() > 3 else "image"
            
            # Create prompt
            prompts = [(prompt, data)]
            print(prompt)
            # Create generator
            gen_cfg = dataclass_from_dict(
                PackedCausalTransformerGeneratorArgs,
                {"temperature": temperature, "top_p": top_p, "top_k": top_k},
                strict=False,
            )
            generator = PackedCausalTransformerGenerator(gen_cfg, self.model, self.tokenizer)
            
            # Run generation
            start_time = time.time() 
            generation, _, _ = generator.generate(prompts)
            end_time = time.time()
            print(generation)
            for gen in generation:
                generated_texts.append(gen)
            
        
        self.clear_temp_files()
                
        return generated_texts
    
    def batch_generate(self,
                      media_paths: List[str],
                      questions: List[str],
                      media_types: List[str],
                      num_frames: int = 32,
                      num_tiles: int = 36,
                      temperature: float = 0.0,
                      top_p: float = None,
                      top_k: int = None,
                      max_new_tokens: int = 256) -> List[str]:
        """
        Batch generation for multiple media files and questions
        """
        assert len(media_paths) == len(questions) == len(media_types), "Inputs must have the same length"
        
        prompts = []
        for path, question, media_type in zip(media_paths, questions, media_types):
            if media_type == "image":
                transform = get_image_transform(
                    vision_input_type=("vanilla" if num_tiles == 1 else self.config.data.vision_input_type),
                    image_res=self.model.vision_model.image_size,
                    max_num_tiles=num_tiles,
                )
                image = Image.open(path).convert("RGB")
                image, _ = transform(image)
                prompts.append((question, image))
            elif media_type == "video":
                transform = get_video_transform(
                    image_res=self.model.vision_model.image_size,
                )
                video_info = (path, num_frames, None, None, None)
                frames, _ = transform(video_info)
                prompts.append((question, frames))
            else:
                raise NotImplementedError(f"Media type {media_type} not supported")
        
        # Create generator
        gen_cfg = dataclass_from_dict(
            PackedCausalTransformerGeneratorArgs,
            {"temperature": temperature, "top_p": top_p, "top_k": top_k},
            strict=False,
        )
        generator = PackedCausalTransformerGenerator(gen_cfg, self.model, self.tokenizer)
        
        # Run generation
        generation, _, _ = generator.generate(prompts)
        
        return generation