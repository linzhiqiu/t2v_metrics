import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Union
import copy
from decord import VideoReader, cpu
import requests
import sys
import warnings


from .vqa_model import VQAScoreModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

warnings.filterwarnings("ignore")
LLAVA_VIDEO_MODELS = {
    'llava-video-7b': {
        'model': {
            'path': "lmms-lab/LLaVA-Video-7B-Qwen2",
            'conversation': 'qwen_1_5',
        },
    },
   'llava-video-72B': {
        'model': {
            'path': "lmms-lab/LLaVA-Video-72B-Qwen2",
            'conversation': 'qwen_1_5',
        },
    }
}

class LLaVAVideoModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self, 
                 model_name='llava-video-7B',
                 device='cuda',
                 cache_dir=None):
        assert model_name in LLAVA_VIDEO_MODELS, f"Model {model_name} not found in LLAVA_VIDEO_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = LLAVA_VIDEO_MODELS[model_name]
        self.conversational_style = self.model_info['model']['conversation']
        self.load_model()
        

    def load_model(self):
        model_path = self.model_info['model']['path']
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", torch_dtype='bfloat16', device_map="auto", attn_implementation='flash_attention_2')
        self.model.eval()

    def load_images(self, video_path, max_frames_num,fps=1,force_sample=False):
        if video_path.startswith(("http://", "https://")):
            raise NotImplementedError("Web link image/video inputs are not yet supported for this model. Please use a local path, or otherwise, make a Github issue request if this feature is necessary.")
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).numpy()
        # import pdb;pdb.set_trace()
        return spare_frames,frame_time,video_time

    def forward(self,
                images: List[str],
                texts: List[str],
                num_frames: int=16,
                question_template: str = "Does this image show \"{}\"? Answer the question with Yes or No",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(images) == len(texts), "Number of paths and texts must match"
    
        lm_probs = []
        for path, text in zip(images, texts):
            if not path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f'This model supports only video inference. {path} is invalid')
                lm_probs.append(0.0)
            else:
                video,frame_time,video_time = self.load_images(path, 64, 1, force_sample=True)
                video = self.processor.preprocess(video, return_tensors='pt')["pixel_values"].cuda().half()
                video = [video]

                question = question_template.format(text)
                answer = answer_template.format(text)

                question = self.format_question(question, video_time, video, frame_time)

                input_ids = tokenizer_image_token(question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model.generate(
                        input_ids,
                        images=video,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=1,
                        modalities=["video"],
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
            num_frames: int = 4,
            max_new_tokens: int = 256,) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        generated_outputs = []
        for path, text in zip(images, texts):
            if not path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f'This model supports only video inference. {path} is invalid')
                generated_outputs.append("")
            else:
                video, frame_time, video_time = self.load_images(path, 64, 1, force_sample=True)
                video = self.processor.preprocess(video, return_tensors='pt')["pixel_values"].cuda().half()
                video = [video]

                # Since this is generate, we'll use the text directly as the question
                question = self.format_question(text, video_time, video, frame_time)

                input_ids = tokenizer_image_token(question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model.generate(
                        input_ids,
                        images=video,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=max_new_tokens,  # Use the passed max_new_tokens
                        modalities=["video"],
                        return_dict_in_generate=True,
                    )
                
                # Decode the generated tokens
                generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                generated_outputs.append(generated_text.strip())

        return generated_outputs

    def format_question(self, question, video_time, video, frame_time):
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."

        conv = copy.deepcopy(conv_templates[self.conversational_style])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n" + question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
