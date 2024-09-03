import os
import torch
import numpy as np
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from decord import VideoReader, cpu
import torch.nn.functional as F
from PIL import Image

from .vqa_model import VQAScoreModel

INTERNVIDEO2_MODELS = {
    'internvideo2-chat-8b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVideo2-Chat-8B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'path': 'OpenGVLab/InternVideo2-Chat-8B',
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
        },
    },
    'internvideo2-chat-8b-hd': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVideo2_chat_8B_HD',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'path': 'OpenGVLab/InternVideo2_chat_8B_HD',
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
        },
    },
}

class InternVideo2Model(VQAScoreModel):
    def __init__(self,
                 model_name='internvideo2-chat-8b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in INTERNVIDEO2_MODELS, f"Model {model_name} not found in INTERNVIDEO2_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = INTERNVIDEO2_MODELS[model_name]
        self.is_hd = 'hd' in model_name
        self.load_model()

    def load_model(self):
        model_path = self.model_info['model']['path']
        tokenizer_path = self.model_info['tokenizer']['path']
        
        self.model = AutoModel.from_pretrained(
            model_path, 
            **self.model_info['model']
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **self.model_info['tokenizer']
        )

    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(self, video_path, num_segments=8, resolution=224, hd_num=6):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = self.get_index(num_frames, num_segments)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        if self.is_hd:
            transform = T.Compose([
                T.Lambda(lambda x: x.float().div(255.0)),
                T.Normalize(mean, std)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda x: x.float().div(255.0)),
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.Normalize(mean, std)
            ])
        
        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        frames = transform(frames)
        
        if self.is_hd:
            frames = self.HD_transform_no_padding(frames, image_size=resolution, hd_num=hd_num)
            T_, C, H, W = frames.shape
            sub_img = frames.reshape(
                1, T_, 3, H//resolution, resolution, W//resolution, resolution
            ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()
            glb_img = F.interpolate(
                frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
            ).to(sub_img.dtype).unsqueeze(0)
            frames = torch.cat([sub_img, glb_img]).unsqueeze(0)
        
        return frames

    def HD_transform_no_padding(self, frames, image_size=224, hd_num=6, fix_ratio=(2,1)):
        _, _, orig_height, orig_width = frames.shape
        aspect_ratio = orig_width / orig_height
        target_aspect_ratio = fix_ratio
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        resized_frame = F.interpolate(
            frames, size=(target_height, target_width),
            mode='bicubic', align_corners=False
        )
        return resized_frame

    def process_image(self, image_path, resolution=224):
        if image_path.lower().endswith('.npy'):
            image = np.load(image_path)
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image = Image.open(image_path).convert('RGB')
            image = T.ToTensor()(image)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        if self.is_hd:
            image = F.interpolate(image.unsqueeze(0), size=(resolution*2, resolution), mode='bicubic', align_corners=False).squeeze(0)
        else:
            image = F.interpolate(image.unsqueeze(0), size=(resolution, resolution), mode='bicubic', align_corners=False).squeeze(0)
        
        # Adjust dimensions to match the model's expected input
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions
        if self.is_hd:
            sub_img = image.reshape(1, 1, 3, 2, resolution, 1, resolution).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, 1, 3, resolution, resolution)
            glb_img = F.interpolate(image.squeeze(1), size=(resolution, resolution), mode='bicubic', align_corners=False).unsqueeze(0).unsqueeze(0)
            image = torch.cat([sub_img, glb_img], dim=0).unsqueeze(0)
        
        return image

    def load_images(self, paths: List[str], num_segments: int = 8) -> List[torch.Tensor]:
        processed_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                processed_data.append(self.load_video(path, num_segments=num_segments))
            else:  # Image file or .npy file
                processed_data.append(self.process_image(path))
        return processed_data

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = "Does this figure show \"{}\"? Please answer yes or no.",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        processed_data = self.load_images(paths)

        lm_probs = []
        for data, question in zip(processed_data, questions):
            data = data.to(self.device)
            chat_history = []
            response, _ = self.model.chat(
                self.tokenizer, 
                '', 
                question, 
                media_type='video' if data.dim() == 6 else 'image',  # 6D for video, 5D for image
                media_tensor=data, 
                chat_history=chat_history, 
                return_history=True,
                generation_config={'do_sample': False, 'output_scores': True, 'return_dict_in_generate': True}
            )

            # Assuming the chat method now returns logit scores
            scores = response.scores[0]  # This line might need adjustment based on the actual output
            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.tokenizer.encode("Yes")[0]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)

        return torch.tensor(lm_probs)