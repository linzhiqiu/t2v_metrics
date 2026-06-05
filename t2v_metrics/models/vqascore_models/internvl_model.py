import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Dict
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu

from .vqa_model import VQAScoreModel
from .fastchat_utils import get_conv_template

import sys

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

INTERNVL_MODELS = {
    # InternVL3 Models
    'internvl3-8b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3-8B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3-8B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3-14b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3-14B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3-14B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3-78b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3-78B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3-78B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },

    # InternVL3.5 Models
    'internvl3.5-1b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3_5-1B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3_5-1B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3.5-2b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3_5-2B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3_5-2B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3.5-4b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3_5-4B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3_5-4B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3.5-8b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3_5-8B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3_5-8B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3.5-14b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3_5-14B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3_5-14B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3.5-30b-a3b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3_5-30B-A3B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3_5-30B-A3B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3.5-38b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3_5-38B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3_5-38B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
    'internvl3.5-241b-a28b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL3_5-241B-A28B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL3_5-241B-A28B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn':  False,
            'trust_remote_code': True,
            'device_map': 'auto',
        },
    },
}


class InternVLModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True

    def __init__(self,
                 model_name='internvl3.5-8b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in INTERNVL_MODELS, \
            f"Model {model_name} not found in INTERNVL_MODELS"
        self.model_name = model_name
        self.device     = device
        self.cache_dir  = cache_dir
        self.model_info = INTERNVL_MODELS[model_name]
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            **self.model_info['model']
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_info['tokenizer']['path'],
            **self.model_info['tokenizer']
        )
        self.device = next(self.model.parameters()).device
        # ------------------------------------------------------------------
        # Image / video preprocessing  (same pipeline as official examples)
        # ------------------------------------------------------------------

    def build_transform(self, input_size=448):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio      = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_ar   = ratio[0] / ratio[1]
            ratio_diff  = abs(aspect_ratio - target_ar)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio      = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_ar     = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width  = image_size * target_ar[0]
        target_height = image_size * target_ar[1]
        blocks        = target_ar[0] * target_ar[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width  // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width  // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        if image_file.lower().endswith('.npy'):
            np_array = np.load(image_file)
            if np_array.ndim == 3:
                image = Image.fromarray(np_array.astype('uint8'), 'RGB')
            else:
                raise ValueError(f"Unexpected NumPy shape in {image_file}")
        else:
            image = Image.open(image_file).convert('RGB')

        transform = self.build_transform(input_size)
        tiles     = self.dynamic_preprocess(image, image_size=input_size,
                                             use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(t) for t in tiles])
        return pixel_values

    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr        = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps       = float(vr.get_avg_fps())
        transform = self.build_transform(input_size)

        if bound:
            start, end = bound
        else:
            start, end = -100000, 100000
        start_idx    = max(0, round(start * fps))
        end_idx      = min(round(end * fps), max_frame)
        seg_size     = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])

        pixel_values_list, num_patches_list = [], []
        for frame_index in frame_indices:
            img    = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            tiles  = self.dynamic_preprocess(img, image_size=input_size,
                                              use_thumbnail=True, max_num=max_num)
            pv     = torch.stack([transform(t) for t in tiles])
            num_patches_list.append(pv.shape[0])
            pixel_values_list.append(pv)

        return torch.cat(pixel_values_list), num_patches_list

    def load_images(self, paths: List[str], num_frames: int = 16):
        processed_data   = []
        num_patches_list = []
        for path in paths:
            if path.startswith(("http://", "https://")):
                raise NotImplementedError(
                    "URL inputs are not supported. Please use a local path."
                )
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                pv, np_list = self.load_video(path, num_segments=num_frames)
                processed_data.append(pv)
                num_patches_list.append(np_list)
            else:
                pv = self.load_image(path)
                processed_data.append(pv)
                num_patches_list.append(pv.shape[0])
        return processed_data, num_patches_list

    # ------------------------------------------------------------------
    # Shared helper: build tokenized query from prompt + pixel patches
    # ------------------------------------------------------------------

    def _build_inputs(self, pixel_values, num_patches, prompt):
        """
        Replicate InternVL's chat tokenization so we can call model.generate
        directly and recover output_scores for VQAScore computation.
        """
        IMG_START_TOKEN   = '<img>'
        IMG_END_TOKEN     = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)

        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        patches = num_patches if isinstance(num_patches, list) else [num_patches]
        for patch in patches:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.model.num_image_token * patch
                + IMG_END_TOKEN
            )
            query = query.replace('<image>', image_tokens, 1)

        model_inputs  = self.tokenizer(query, return_tensors='pt')
        input_ids     = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)

        return input_ids, attention_mask, eos_token_id

    # ------------------------------------------------------------------
    # forward — VQAScore
    # ------------------------------------------------------------------

    def forward(self,
                paths: List[str],
                texts: List[str],
                num_frames: int = 16,
                question_template: str = 'Does this figure show "{}"? Please answer Yes or No.',
                answer_template: str = 'Yes') -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(t) for t in texts]
        answers   = [answer_template.format(t)   for t in texts]
        processed_data, num_patches_list = self.load_images(paths, num_frames)

        lm_probs = []
        for data, question, num_patches, answer in zip(
            processed_data, questions, num_patches_list, answers
        ):
            if isinstance(num_patches, list):
                prefix = ''.join(
                    [f'Frame{i+1}: <image>\n' for i in range(len(num_patches))]
                )
                prompt = prefix + question
            else:
                prompt = '<image>\n' + question

            pixel_values = data.to(self.device).to(self.model.dtype)
            input_ids, attention_mask, eos_token_id = self._build_inputs(
                pixel_values, num_patches, prompt
            )

            generation_config = dict(
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                eos_token_id=eos_token_id,
            )

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config,
                )

            probs        = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
            yes_token_id = self.tokenizer.encode(answer)[0]
            lm_probs.append(probs[0, yes_token_id].item())

        return torch.tensor(lm_probs)

    # ------------------------------------------------------------------
    # generate — free-form text generation
    # ------------------------------------------------------------------

    def generate(self,
                 images: List[str],
                 texts: List[str],
                 num_frames: int = 16,
                 max_new_tokens: int = 1024,
                 do_sample: bool = False,
                 temperature: float = 0.0) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"

        processed_data, num_patches_list = self.load_images(images, num_frames)

        if do_sample is None:
            do_sample = (temperature > 0)

        gen_outputs = []
        for data, question, num_patches in zip(processed_data, texts, num_patches_list):
            if isinstance(num_patches, list):
                prefix = ''.join(
                    [f'Frame{i+1}: <image>\n' for i in range(len(num_patches))]
                )
                prompt = prefix + question
            else:
                prompt = '<image>\n' + question

            pixel_values = data.to(self.device).to(self.model.dtype)
            input_ids, attention_mask, eos_token_id = self._build_inputs(
                pixel_values, num_patches, prompt
            )

            generation_config = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
            )
            if do_sample and temperature > 0:
                generation_config['temperature'] = temperature

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config,
                )

            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            gen_outputs.append(response)

        return gen_outputs