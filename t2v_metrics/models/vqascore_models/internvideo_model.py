import os
import torch
import numpy as np
import decord
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms
from torchvision.transforms import PILToTensor
from decord import VideoReader, cpu
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from typing import List, TypedDict, Union, Optional
decord.bridge.set_bridge("torch")

from .vqa_model import VQAScoreModel

INTERNVIDEO2_MODELS = {
    'internvideo2-chat-8b': {
        'tokenizer': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVideo2-Chat-8B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVideo2-Chat-8B',
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
        },
    },
    'internvideo2-chat-8b-hd': {
        'tokenizer': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVideo2_chat_8B_HD',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVideo2_chat_8B_HD',
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
        },
    },
    'internvideo2-chat-8b-internlm': {
        'tokenizer': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVideo2_Chat_8B_InternLM2_5',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVideo2_Chat_8B_InternLM2_5',
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
        },
    },
}

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMAGE_TOKEN = "[IMAGETOKEN]"
DEFAULT_VIDEO_TOKEN = "[VIDEOTOKEN]"

DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"
DEFAULT_VID_PLACEHOLDER = "[<VID_PLH>]"
class InternVideo2Model(VQAScoreModel):
    video_mode = "direct"
    allows_image = False
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
        
        self.model = AutoModel.from_pretrained(
            **self.model_info['model']
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            **self.model_info['tokenizer']
        )

    def load_video(self, video_path, num_segments=8, resolution=224, hd_num=6):
        if self.model_name == 'internvideo2-chat-8b':
            return self.load_video_chat(video_path, num_segments, return_msg=False)
        elif self.model_name == 'internvideo2-chat-8b-hd':
            return self.load_video_chat_hd(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)

    def process_image(self, path, resolution=224):
        if self.model_name == 'internvideo2-chat-8b':
            return self.process_image_chat(path)
        elif self.model_name == 'internvideo2-chat-8b-hd':
            return self.process_image_chat_hd(path, resolution)
        

    def load_images(self, paths: List[str], num_segments: int = 8, resolution: int = 224, hd_num: int = 6) -> List[torch.Tensor]:
        processed_data = []
        for path in paths:
            if path.startswith(("http://", "https://")):
                raise NotImplementedError("Web link image/video inputs are not yet supported for this model. Please use a local path, or otherwise, make a Github issue request if this feature is necessary.")
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                processed_data.append(self.load_video(path, num_segments=num_segments, resolution=resolution, hd_num=hd_num))
            else:  # Image file or .npy file
                print('This model does not yet support image inference.')
                exit()
                processed_data.append(self.process_image(path, resolution))
        return processed_data

    def forward(self,
                paths: List[str],
                texts: List[str],
                num_frames: int=8,
                question_template: str = "Does this figure show \"{}\"? Please answer yes or no.",
                answer_template: str = "Yes",
                num_segments: int = None,
                resolution: int = None,
                hd_num: int = None) -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        processed_data = self.load_images(paths, num_segments=num_frames)

        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
            data = data.to(self.device)
            chat_history = []
            media_type='video' if data.dim() == 6 else 'image'  # 6D for video, 5D for image
            instruction=None
            msg = ''
            user_prompt = question
            media_tensor = data
            if self.model_name == 'internvideo2-chat-8b':
                input_ids, attention_masks, labels = [], [], []

                conversation = ""
                if instruction:
                    conversation += instruction
                conversation += (
                            "[INST]" + " "
                        )

                if media_type == 'image':
                    conversation +=( "<Image>" + IMG_TOKEN + "</Image>")#*ilen
                else:
                    conversation += ("<Video>" + VID_TOKEN + "</Video>")#*ilen


                conversation += (
                            msg.rstrip() + "[/INST]"
                        )

                for q,a in chat_history:
                    conversation += (" [INST] " + q + " [/INST]")
                    conversation += (a + "</s>")

                conversation += (" [INST] " + user_prompt + " [/INST]")
                conversation += ("")


                total_len = 0
                indexs = []
                tokenized = self.build_input_ids_chat(
                    self.tokenizer,
                    conversation,
                    max_length=248,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_tensors='pt'
                )
                if media_type == 'image':
                    response = self.generate_caption_chat(
                        tokenized['input_ids'].unsqueeze(0).to(self.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                        image_idx = tokenized['index'].unsqueeze(0),
                        image = media_tensor.unsqueeze(0), 
                       )
                else:
                    response = self.generate_caption_chat(
                        tokenized['input_ids'].unsqueeze(0).to(self.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                        video_idx = tokenized['index'].unsqueeze(0),
                        video = media_tensor.unsqueeze(0), 
                       )
            elif self.model_name == 'internvideo2-chat-8b-hd':
                answer_prompt = None
                conversation = ""
                if instruction:
                    conversation += instruction
                conversation += (
                            "[INST]" + " "
                        )

                if media_type == 'image':
                    ilen = media_tensor.shape[0]
                    conversation +=( "<Image>" + IMG_TOKEN + "</Image>")*ilen
                else:
                    ilen = media_tensor.shape[1]
                    conversation += ("<Video>" + VID_TOKEN + "</Video>")*ilen


                conversation += (
                            msg.rstrip() + "[/INST]"
                        )

                for q,a in chat_history:
                    conversation += (" [INST] " + q + " [/INST]")
                    conversation += (a + "</s>")

                conversation += (" [INST] " + user_prompt + " [/INST]")
                if answer_prompt:
                    conversation += (answer_prompt)
                else:
                    conversation += ("")


                total_len = 0
                indexs = []
                # ilen = media_tensor.shape[1]

                # conversation = ""
                # if instruction:
                #     conversation += instruction
                # conversation += (
                #             "[INST]" + " "
                #         )

                # if media_type == 'image':
                #     conversation +=( "<Image>" + IMG_TOKEN + "</Image>")*ilen
                # else:
                #     conversation += ("<Video>" + VID_TOKEN + "</Video>")*ilen


                # conversation += (
                #             msg.rstrip() + "[/INST]"
                #         )

                # for q,a in chat_history:
                #     conversation += (" [INST] " + q + " [/INST]")
                #     conversation += (a + "</s>")

                # conversation += (" [INST] " + user_prompt + " [/INST]")
                # conversation += ("")


                # total_len = 0
                # indexs = []
                tokenized = self.build_input_ids_chat_hd(
                    self.tokenizer,
                    conversation,
                    max_length=1024,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_tensors='pt'
                )
                if media_type == 'image':
                    response = self.generate_caption_chat_hd(
                        tokenized['input_ids'].unsqueeze(0).to(self.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                        image_idx = tokenized['index'].unsqueeze(0),
                        image = media_tensor,
                        instruction=[instruction]* ilen if instruction else None,
                 
                    )
                else:
                    response = self.generate_caption_chat_hd(
                        tokenized['input_ids'].unsqueeze(0).to(self.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                        video_idx = tokenized['index'].unsqueeze(0),
                        video = media_tensor, 
                        instruction=[instruction]* ilen if instruction else None,
                    )
   
            scores = response.scores[0] 

            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.tokenizer.encode(answer)[1]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)

        return torch.tensor(lm_probs)

    def generate(self,
                    images: List[str],
                    texts: List[str],
                    num_frames: int=8,
                    resolution: int = None,
                    hd_num: int = None) -> torch.Tensor:
        assert len(images) == len(texts), "Number of paths and texts must match"

        questions = texts
        processed_data = self.load_images(images, num_segments=num_frames)

        gen_outputs = []
        for data, question in zip(processed_data, questions):
            data = data.to(self.device)
            chat_history = []
            media_type='video' if data.dim() == 6 else 'image'  # 6D for video, 5D for image
            instruction=None
            msg = ''
            user_prompt = question
            media_tensor = data
            if self.model_name == 'internvideo2-chat-8b':
                input_ids, attention_masks, labels = [], [], []

                conversation = ""
                if instruction:
                    conversation += instruction
                conversation += (
                            "[INST]" + " "
                        )

                if media_type == 'image':
                    conversation +=( "<Image>" + IMG_TOKEN + "</Image>")#*ilen
                else:
                    conversation += ("<Video>" + VID_TOKEN + "</Video>")#*ilen


                conversation += (
                            msg.rstrip() + "[/INST]"
                        )

                for q,a in chat_history:
                    conversation += (" [INST] " + q + " [/INST]")
                    conversation += (a + "</s>")

                conversation += (" [INST] " + user_prompt + " [/INST]")
                conversation += ("")


                total_len = 0
                indexs = []
                tokenized = self.build_input_ids_chat(
                    self.tokenizer,
                    conversation,
                    max_length=248,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_tensors='pt'
                )
                if media_type == 'image':
                    response = self.generate_caption_chat(
                        tokenized['input_ids'].unsqueeze(0).to(self.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                        image_idx = tokenized['index'].unsqueeze(0),
                        image = media_tensor.unsqueeze(0), 
                        max_new_tokens=256
                    )
                else:
                    response = self.generate_caption_chat(
                        tokenized['input_ids'].unsqueeze(0).to(self.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                        video_idx = tokenized['index'].unsqueeze(0),
                        video = media_tensor.unsqueeze(0), 
                        max_new_tokens=256
                    )
            elif self.model_name == 'internvideo2-chat-8b-hd':
                answer_prompt = None
                conversation = ""
                if instruction:
                    conversation += instruction
                conversation += (
                            "[INST]" + " "
                        )

                if media_type == 'image':
                    ilen = media_tensor.shape[0]
                    conversation +=( "<Image>" + IMG_TOKEN + "</Image>")*ilen
                else:
                    ilen = media_tensor.shape[1]
                    conversation += ("<Video>" + VID_TOKEN + "</Video>")*ilen


                conversation += (
                            msg.rstrip() + "[/INST]"
                        )

                for q,a in chat_history:
                    conversation += (" [INST] " + q + " [/INST]")
                    conversation += (a + "</s>")

                conversation += (" [INST] " + user_prompt + " [/INST]")
                if answer_prompt:
                    conversation += (answer_prompt)
                else:
                    conversation += ("")


                total_len = 0
                indexs = []
                # ilen = media_tensor.shape[1]

                # conversation = ""
                # if instruction:
                #     conversation += instruction
                # conversation += (
                #             "[INST]" + " "
                #         )

                # if media_type == 'image':
                #     conversation +=( "<Image>" + IMG_TOKEN + "</Image>")*ilen
                # else:
                #     conversation += ("<Video>" + VID_TOKEN + "</Video>")*ilen


                # conversation += (
                #             msg.rstrip() + "[/INST]"
                #         )

                # for q,a in chat_history:
                #     conversation += (" [INST] " + q + " [/INST]")
                #     conversation += (a + "</s>")

                # conversation += (" [INST] " + user_prompt + " [/INST]")
                # conversation += ("")


                # total_len = 0
                # indexs = []
                tokenized = self.build_input_ids_chat_hd(
                    self.tokenizer,
                    conversation,
                    max_length=1024,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_tensors='pt'
                )
                if media_type == 'image':
                    response = self.generate_caption_chat_hd(
                        tokenized['input_ids'].unsqueeze(0).to(self.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                        image_idx = tokenized['index'].unsqueeze(0),
                        image = media_tensor,
                        instruction=[instruction]* ilen if instruction else None,
                        max_new_tokens=256
                
                    )
                else:
                    response = self.generate_caption_chat_hd(
                        tokenized['input_ids'].unsqueeze(0).to(self.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                        video_idx = tokenized['index'].unsqueeze(0),
                        video = media_tensor, 
                        instruction=[instruction]* ilen if instruction else None,
                        max_new_tokens=256
                    )
       
    
            tokens = response.sequences[0] 
            text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

            gen_outputs.append(text)

        return gen_outputs
        
    # **
    # **
    # Helper Functions:
    # **
    # **

    def load_video_chat(self, video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = self.get_index(num_frames, num_segments)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std)
        ])

        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        frames = transform(frames)

        T_, C, H, W = frames.shape
            
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return frames, msg
        else:
            return frames
        
    def load_video_chat_hd(self, video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = self.get_index(num_frames, num_segments)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize(mean, std)
        ])

        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)

        if padding:
            frames = self.HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
        else:
            frames = self.HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

        frames = transform(frames)
        T_, C, H, W = frames.shape

        sub_img = frames.reshape(
            1, T_, 3, H//resolution, resolution, W//resolution, resolution
        ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

        glb_img = F.interpolate(
            frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
        ).to(sub_img.dtype).unsqueeze(0)

        frames = torch.cat([sub_img, glb_img]).unsqueeze(0)

        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return frames, msg
        else:
            return frames

    # *
    # HD Video Loading Helper Functions
    # * 

    def HD_transform_padding(self, frames, image_size=224, hd_num=6):
        def _padding_224(frames):
            _, _, H, W = frames.shape
            tar = int(np.ceil(H / 224) * 224)
            top_padding = (tar - H) // 2
            bottom_padding = tar - H - top_padding
            left_padding = 0
            right_padding = 0

            padded_frames = F.pad(
                frames,
                pad=[left_padding, right_padding, top_padding, bottom_padding],
                mode='constant', value=255
            )
            return padded_frames

        _, _, H, W = frames.shape
        trans = False
        if W < H:
            frames = frames.flip(-2, -1)
            trans = True
            width, height = H, W
        else:
            width, height = W, H

        ratio = width / height
        scale = 1
        while scale * np.ceil(scale / ratio) <= hd_num:
            scale += 1
        scale -= 1
        new_w = int(scale * image_size)
        new_h = int(new_w / ratio)

        resized_frames = F.interpolate(
            frames, size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        )
        padded_frames = _padding_224(resized_frames)

        if trans:
            padded_frames = padded_frames.flip(-2, -1)

        return padded_frames

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


    def HD_transform_no_padding(self, frames, image_size=224, hd_num=6, fix_ratio=(2,1)):
        min_num = 1
        max_num = hd_num
        _, _, orig_height, orig_width = frames.shape
        aspect_ratio = orig_width / orig_height

        # calculate the existing video aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        if fix_ratio:
            target_aspect_ratio = fix_ratio
        else:
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the frames
        resized_frame = F.interpolate(
            frames, size=(target_height, target_width),
            mode='bicubic', align_corners=False
        )
        return resized_frame
    def process_image_chat(self, image_path, resolution=224):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        d_hid = self.model.vision_encoder.encoder.img_pos_embed.shape[-1]  # Get embedding dimension

        if image_path.lower().endswith('.npy'):
            img = np.load(image_path)
            img = torch.from_numpy(img)
        else:
            img = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((resolution, resolution), 
                            interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            img = transform(img)

        img = img.unsqueeze(0).unsqueeze(0)  # Add batch and frame dimensions
        
        # For single image, we can use ckpt_num_frame=-1 to skip interpolation
        n_position = (resolution//16)**2
        new_pos_emb = self.get_sinusoid_encoding_table(
            n_position=n_position,
            d_hid=d_hid,
            ckpt_num_frame=-1,  # Skip frame interpolation for single image
            cur_frame=1  # Single frame
        )
        self.model.vision_encoder.encoder.img_pos_embed = new_pos_emb

        return img.to(self.device)

    def process_image_chat_hd(self, image_path, resolution=224, hd_num=12, padding=False):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if image_path.lower().endswith('.npy'):
            img = np.load(image_path)
            img = torch.from_numpy(img)
        else:
            img = Image.open(image_path).convert('RGB')
            img = PILToTensor()(img)
        
        # Match authors' implementation
        img = img.float().div(255.0).unsqueeze(0)  # Adding batch dimension first

        if padding:
            img = self.HD_transform_padding(img, image_size=resolution, hd_num=hd_num)
        else:
            img = self.HD_transform_no_padding(img, image_size=resolution, hd_num=hd_num)

        transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        
        img = transform(img).unsqueeze(0)  # Final unsqueeze for model input
        return img.to(self.device)
    
    def get_sinusoid_encoding_table(self, n_position, d_hid, ckpt_num_frame=-1, cur_frame=12): 
        ''' Sinusoid position encoding table ''' 
        # TODO: make it with torch instead of numpy 
        def get_position_angle_vec(position): 
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
        
        if ckpt_num_frame != -1 and ckpt_num_frame != cur_frame:
            logger.info(f"Interpolate position embedding")
            logger.info(f"Testing frame: {cur_frame}")
            logger.info(f"Checkpoint frame: {ckpt_num_frame}")

            T = ckpt_num_frame # checkpoint frame
            new_T = cur_frame # testing frame
            n_position = n_position // new_T * T # generate checkpoint position embedding
            sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
            sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
            # interpolate
            P = int((n_position // T) ** 0.5)
            C = d_hid
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
            sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
            sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
            sinusoid_table = sinusoid_table.flatten(1, 3)
            return sinusoid_table
        else:
            sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
            return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0) 
    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets


    # *
    # Chat Helper Functions: Note, there is probably a more efficient way to do this, but this is the most foolproof way.
    # *

    def generate_caption_chat_hd(
        self,
        input_ids,
        attention_mask,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=1,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
        instruction=None
    ):
       
        text_embeds = self.model.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx, video_idx=video_idx,instruction=instruction)
        
        # outputs = self.lm.generate(
        #     inputs_embeds=text_embeds,
        #     attention_mask=attention_mask,
        #     num_beams=num_beams,
        #     max_new_tokens=max_new_tokens,
        #     do_sample=do_sample,
        #     min_length=1,
        #     top_p=top_p,
        #     top_k=top_k,
        #     temperature=temperature,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        # )
        # Ignoring InternVideo2's default parameters
        outputs = self.model.lm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True, 
            return_dict_in_generate=True
        )

        return outputs
    def build_input_ids_chat_hd(
            self, 
            tokenizer, 
            conversation,
            max_length,
            add_special_tokens,
            truncation,
            image = None, 
            video = None, 
            padding = "longest", 
            return_tensors = "pt",
            image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
            video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        input_ids = []
        indexs = []
        attention_mask = []
        start, total_len = 0, 0
        while True:
            index1 = conversation.find(image_placeholder, start)
            index2 = conversation.find(video_placeholder, start)
            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)
                assert index != -1
            if index == -1:
                inputs = tokenizer(conversation[start:], max_length=max_length-total_len, truncation=truncation, padding=padding, return_tensors=return_tensors)
            else:
                inputs = tokenizer(conversation[start:index], max_length=max_length,  truncation=truncation, padding='longest', return_tensors=return_tensors)
            
            input_ids += inputs.input_ids
            attention_mask += inputs.attention_mask
            total_len += inputs.input_ids[0].shape[0]
            indexs += torch.zeros_like(inputs.input_ids)
            
            if index != -1:
                input_ids += [torch.zeros(96).long()]
                attention_mask += [torch.ones(96).long()]
                indexs += [torch.ones(96)]
            
            if index == -1:
                return {
                    'input_ids': torch.cat(input_ids),
                    'attention_mask': torch.cat(attention_mask),
                    'index': torch.cat(indexs).to(torch.bool),
                }
            start = index + len(DEFAULT_IMG_PLACEHOLDER)

    def generate_caption_chat(
        self,
        input_ids,
        attention_mask,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=1,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
    ):
        
        text_embeds = self.model.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx, video_idx=video_idx)
        # outputs = self.lm.generate(
        #     inputs_embeds=text_embeds,
        #     attention_mask=attention_mask,
        #     num_beams=num_beams,
        #     max_new_tokens=max_new_tokens,
        #     do_sample=do_sample,
        #     min_length=1,
        #     top_p=top_p,
        #     top_k=top_k,
        #     temperature=temperature,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        # )
        # Ignoring InternVideo2's default parameters
        outputs = self.model.lm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True, 
            return_dict_in_generate=True
        )
        return outputs
    
    def build_input_ids_chat(
            self, 
            tokenizer, 
            conversation,
            max_length,
            add_special_tokens,
            truncation,
            image = None, 
            video = None, 
            padding = "longest", 
            return_tensors = "pt",
            image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
            video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        input_ids = []
        indexs = []
        attention_mask = []
        start, total_len = 0, 0
        while True:
            index1 = conversation.find(image_placeholder, start)
            index2 = conversation.find(video_placeholder, start)
            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)
                assert index != -1
            if index == -1:
                inputs = tokenizer(conversation[start:], max_length=max_length-total_len, truncation=truncation, padding=padding, return_tensors=return_tensors)
            else:
                inputs = tokenizer(conversation[start:index], max_length=max_length,  truncation=truncation, padding='longest', return_tensors=return_tensors)
            
            input_ids += inputs.input_ids
            attention_mask += inputs.attention_mask
            total_len += inputs.input_ids[0].shape[0]
            indexs += torch.zeros_like(inputs.input_ids)
            
            if index != -1:
                input_ids += [torch.zeros(96).long()]
                attention_mask += [torch.ones(96).long()]
                indexs += [torch.ones(96)]
            
            if index == -1:
                return {
                    'input_ids': torch.cat(input_ids),
                    'attention_mask': torch.cat(attention_mask),
                    'index': torch.cat(indexs).to(torch.bool),
                }
            start = index + len(DEFAULT_IMG_PLACEHOLDER)