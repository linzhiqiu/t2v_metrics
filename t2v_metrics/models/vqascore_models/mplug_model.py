import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu

from .vqa_model import VQAScoreModel

MPLUG_OWL3_MODELS = {
    'mplug-owl3-7b': {
        'tokenizer': {
            'path': 'mPLUG/mPLUG-Owl3-7B-240728',
        },
        'model': {
            'pretrained_model_name_or_path': 'mPLUG/mPLUG-Owl3-7B-240728',
            'attn_implementation': 'flash_attention_2',
            'trust_remote_code': True,
            'torch_dtype': torch.bfloat16,
        },
    },
}

class mPLUGOwl3Model(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='mplug-owl3-7b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in MPLUG_OWL3_MODELS, f"Model {model_name} not found in MPLUG_OWL3_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = MPLUG_OWL3_MODELS[model_name]
        self.load_model()
        

    def load_model(self):
        # model_path = self.model_info['model']['path']
        tokenizer_path = self.model_info['tokenizer']['path']
        
        self.model = AutoModelForCausalLM.from_pretrained(
            # model_path, 
            **self.model_info['model']
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.processor = self.model.init_processor(self.tokenizer)

    def load_images(self, paths: List[str], num_frames: int = 16) -> List[Union[torch.Tensor, List[Image.Image]]]:
        processed_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                video_frames = self.encode_video(path, num_frames)
                processed_data.append(video_frames)
            elif path.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    processed_data.append(image)
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [Image.fromarray(frame.astype('uint8'), 'RGB') for frame in np_array]
                    processed_data.append(frames)
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                processed_data.append(image)
        return processed_data

    def encode_video(self, video_path, max_frames_num):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > max_frames_num:
            frame_idx = uniform_sample(frame_idx, max_frames_num)
        frames = vr.get_batch(frame_idx).numpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        return frames

    def forward(self,
                paths: List[str],
                texts: List[str],
                num_frames: int=16,
                question_template: str = "Does this figure show \"{}\"? Please answer yes or no.",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(paths, num_frames)

        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
            if isinstance(data, list):  # Video
                messages = [
                    {"role": "user", "content": f"<|video|>\n{question}"},
                    {"role": "assistant", "content": ""}
                ]
                inputs = self.processor(messages, images=None, videos=[data])
            else:  # Image
                messages = [
                    {"role": "user", "content": f"<|image|>\n{question}"},
                    {"role": "assistant", "content": ""}
                ]
                inputs = self.processor(messages, images=[data], videos=None)

            inputs.to(self.device)

            with torch.inference_mode():
                image_embeds = self.model.forward_image(inputs['pixel_values'])

                terminators = [self.tokenizer.convert_tokens_to_ids(i) for i in self.model.terminators]

                outputs = self.model.language_model.generate(
                    input_ids=inputs['input_ids'],
                    image_embeds=image_embeds,
                    media_offset=inputs['media_offset'],
                    pad_token_id=0,
                    eos_token_id=terminators,
                    attention_mask=None,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
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
                num_frames: int=16,
                max_new_tokens: int = 256) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"

        processed_data = self.load_images(images, num_frames)

        generated_texts = []
        for data, text in zip(processed_data, texts):
            if isinstance(data, list):  # Video
                messages = [
                    {"role": "user", "content": f"<|video|>\n{text}"},
                    {"role": "assistant", "content": ""}
                ]
                inputs = self.processor(messages, images=None, videos=[data])
            else:  # Image
                messages = [
                    {"role": "user", "content": f"<|image|>\n{text}"},
                    {"role": "assistant", "content": ""}
                ]
                inputs = self.processor(messages, images=[data], videos=None)

            inputs.to(self.device)

            with torch.inference_mode():
                image_embeds = self.model.forward_image(inputs['pixel_values'])

                terminators = [self.tokenizer.convert_tokens_to_ids(i) for i in self.model.terminators]

                outputs = self.model.language_model.generate(
                    input_ids=inputs['input_ids'],
                    image_embeds=image_embeds,
                    media_offset=inputs['media_offset'],
                    pad_token_id=0,
                    eos_token_id=terminators,
                    attention_mask=None,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split('\nassistant\n')[1]
                generated_texts.append(text.strip())

        return generated_texts