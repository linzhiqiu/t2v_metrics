import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu

from .vqa_model import VQAScoreModel
from .fastchat_utils import get_conv_template

INTERNVL2_MODELS = {
    # InternVL2 Models
    'internvl2-1b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2-1B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2-1B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2-2b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2-2B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2-2B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2-4b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2-4B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2-4B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2-8b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2-8B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2-8B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2-26b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2-26B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2-26B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2-40b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2-40B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2-40B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2-llama3-76b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2-Llama3-76B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2-Llama3-76B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    
    # InternVL2.5 Models
    'internvl2.5-1b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2_5-1B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2_5-1B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2.5-2b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2_5-2B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2_5-2B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2.5-4b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2_5-4B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2_5-4B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
        },
    },
    'internvl2.5-8b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2_5-8B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2_5-8B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
            'device_map': 'auto'
        },
    },
    'internvl2.5-26b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2_5-26B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2_5-26B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
            'device_map': 'auto'
        },
    },
    'internvl2.5-38b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2_5-38B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2_5-38B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
            'device_map': 'auto'
        },
    },
    'internvl2.5-78b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2_5-78B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2_5-78B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
            'device_map': 'auto'
        },
    },

    'internvl3-8b': {
    "tokenizer": {
        "path": "OpenGVLab/InternVL3-8B",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "model": {
        "pretrained_model_name_or_path": "OpenGVLab/InternVL3-8B",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "use_flash_attn": True,
        "trust_remote_code": True, 
        'device_map': 'auto'
        },
    },

    'internvl3-14b': {
    "tokenizer": {
        "path": "OpenGVLab/InternVL3-14B",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "model": {
        "pretrained_model_name_or_path": "OpenGVLab/InternVL3-14B",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "use_flash_attn": True,
        "trust_remote_code": True,
        'device_map': 'auto'
        },
    },

    'internvl3-78b': {
    "tokenizer": {
        "path": "OpenGVLab/InternVL3-78B",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "model": {
        "pretrained_model_name_or_path": "OpenGVLab/InternVL3-78B",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "use_flash_attn": True,
        "trust_remote_code": True,
        'device_map': 'auto'
        },
    }
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVL2Model(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='internvl2-8b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in INTERNVL2_MODELS, f"Model {model_name} not found in INTERNVL2_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = INTERNVL2_MODELS[model_name]
        self.load_model()
        

    def load_model(self):
        tokenizer_path = self.model_info['tokenizer']['path']
        # self.model_info['model']['max_memory'] = {
        #         0: "45GB",  # GPU 0
        #         1: "45GB",  # GPU 1
        #         2: "45GB",  # GPU 4
        # }
        
        self.model = AutoModel.from_pretrained(
            **self.model_info['model']
        ).eval()#.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **self.model_info['tokenizer']
        )
        
        # if 'internvl2' in self.model_name:
        self.device = next(self.model.parameters()).device # If there are multiple GPUs put the model on the first parameters GPU
        # elif 'internvl3' in self.model_name:
        #     self.device
    def build_transform(self, input_size=448):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        if image_file.lower().endswith('.npy'):
            np_array = np.load(image_file)
            if np_array.ndim == 3:
                image = Image.fromarray(np_array.astype('uint8'), 'RGB')
            else:
                raise ValueError(f"Unexpected shape for NumPy array in {image_file}")
        else:
            image = Image.open(image_file).convert('RGB')
        
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_video_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        frame_indices = self.get_video_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def load_images(self, paths: List[str], num_frames: int = 16) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        num_patches_list = []
        for path in paths:
            if path.startswith(("http://", "https://")):
                raise NotImplementedError("Web link image/video inputs are not yet supported for this model. Please use a local path, or otherwise, make a Github issue request if this feature is necessary.")
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                video_frames, video_num_patches = self.load_video(path, num_segments=num_frames)
                processed_data.append(video_frames)
                num_patches_list.append(video_num_patches) # You want a list of lists
            else:  # Image file or .npy file
                image_tensor = self.load_image(path)
                processed_data.append(image_tensor)
                num_patches_list.append(image_tensor.shape[0])
        return processed_data, num_patches_list

    def forward(self,
                paths: List[str],
                texts: List[str],
                num_frames: int=16,
                question_template: str = "Does this figure show \"{}\"? Please answer Yes or No.",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data, num_patches_list = self.load_images(paths, num_frames)

        lm_probs = []
        for data, question, num_patches, answer in zip(processed_data, questions, num_patches_list, answers):
            if isinstance(num_patches, list):  # Video
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches))])
                prompt = video_prefix + question
            else:  # Image
                prompt = '<image>\n' + question
                num_patches = [num_patches]

            # Reimplement chat method to allow for outputting scores :(
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # pixel_values = data.to(self.device).to(self.model.dtype)

            # with torch.no_grad():
            #     outputs = self.model.generate(
            #         **inputs,
            #         images=pixel_values,
            #         num_patches_list=[num_patches],
            #         max_new_tokens=1,
            #         do_sample=False,
            #         output_scores=True,
            #         return_dict_in_generate=True,
            #     )

            # Chat Arguments:
            pixel_values = data.to(self.device)
            pixel_values = pixel_values.to(self.model.dtype)
            generation_config = dict(max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True)
            IMG_START_TOKEN='<img>'
            IMG_END_TOKEN='</img>' 
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
            verbose=False
            history=None

            
            assert pixel_values is None or len(pixel_values) == sum(num_patches)
            img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            self.model.img_context_token_id = img_context_token_id

            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)

            history = [] if history is None else history
            for (old_question, old_answer) in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            if verbose and pixel_values is not None:
                image_bs = pixel_values.shape[0]
                print(f'dynamic ViT batch size: {image_bs}')

            for patch in num_patches:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * patch + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            model_inputs = self.tokenizer(query, return_tensors='pt')
            input_ids = model_inputs['input_ids'].to(self.device)
            attention_mask = model_inputs['attention_mask'].to(self.device)
            generation_config['eos_token_id'] = eos_token_id
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )


            scores = outputs.scores[0]

            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.tokenizer.encode(answer)[0] # Option 1

            # Option 2:
            # if answer == "Yes":
            #     yes_token_id = self.tokenizer.encode(" " + answer, add_special_tokens=False)[0]

            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)

        return torch.tensor(lm_probs)
    
    def generate(self,
                images: List[str],
                texts: List[str],
                num_frames: int=16,
                max_new_tokens: int = 256) -> torch.Tensor:
        assert len(images) == len(texts), "Number of paths and texts must match"

        questions = texts
        processed_data, num_patches_list = self.load_images(images, num_frames)

        gen_outputs = []
        for data, question, num_patches in zip(processed_data, questions, num_patches_list):
            if isinstance(num_patches, list):  # Video
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches))])
                prompt = video_prefix + question
            else:  # Image
                prompt = '<image>\n' + question
                num_patches = [num_patches]

            pixel_values = data.to(self.device)
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
            IMG_START_TOKEN='<img>'
            IMG_END_TOKEN='</img>' 
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
            verbose=False
            history=None

            
            assert pixel_values is None or len(pixel_values) == sum(num_patches)
            img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            self.model.img_context_token_id = img_context_token_id

            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

            history = [] if history is None else history
            for (old_question, old_answer) in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            if verbose and pixel_values is not None:
                image_bs = pixel_values.shape[0]
                print(f'dynamic ViT batch size: {image_bs}')

            for patch in num_patches:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * patch + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            model_inputs = self.tokenizer(query, return_tensors='pt')
            input_ids = model_inputs['input_ids'].to(self.model.device)
            attention_mask = model_inputs['attention_mask'].to(self.device)
            generation_config['eos_token_id'] = eos_token_id
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )


            outputs = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
            gen_outputs.append(outputs)

        return gen_outputs
