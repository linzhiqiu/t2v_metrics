from PIL import Image
import os

import torch
from transformers import AutoTokenizer
from ...constants import HF_CACHE_DIR, IMAGE_TOKEN_INDEX

# from moviepy.editor import VideoFileClip


import cv2
import numpy as np
from math import ceil, sqrt

def extract_frames(video_path, num_frames, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the step size to get equally-spaced frames
    step = total_frames // num_frames
    
    for i in range(num_frames):
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        
        # Read the frame
        ret, frame = video.read()
        
        if ret:
            # Save the frame as an image
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, frame)
        else:
            print(f"Failed to extract frame {i}")
    
    # Release the video object
    video.release()

    print(f"Extracted {num_frames} frames from the video.")

# Following functions adapted from: https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/blob/main/llava-multi-images.py
def concatenate_images_vertical(images, dist_images):
    # calc max width from imgs
    width = max(img.shape[1] for img in images)
    # calc total height of imgs + dist between them
    total_height = sum(img.shape[0] for img in images) + dist_images * (len(images) - 1)

    # create new img with calculated dimensions, black bg
    new_img = np.zeros((total_height, width, 3), dtype=np.uint8)

    # init var to track current height pos
    current_height = 0
    for img in images:
        h, w = img.shape[:2]
        # paste img in new_img at current height
        new_img[current_height:current_height+h, :w] = img
        # update current height for next img
        current_height += h + dist_images

    return new_img

def concatenate_images_horizontal(images, dist_images):
    # calc total width of imgs + dist between them
    total_width = sum(img.shape[1] for img in images) + dist_images * (len(images) - 1)
    # calc max height from imgs
    height = max(img.shape[0] for img in images)

    # create new img with calculated dimensions, black bg
    new_img = np.zeros((height, total_width, 3), dtype=np.uint8)

    # init var to track current width pos
    current_width = 0
    for img in images:
        h, w = img.shape[:2]
        # paste img in new_img at current width
        new_img[:h, current_width:current_width+w] = img
        # update current width for next img
        current_width += w + dist_images

    return new_img

def concatenate_images_grid(images, dist_images, output_size):
    num_images = len(images)
    # calc grid size based on amount of input imgs
    grid_size = max(2, ceil(sqrt(num_images)))

    cell_width = (output_size[0] - dist_images * (grid_size - 1)) // grid_size
    cell_height = (output_size[1] - dist_images * (grid_size - 1)) // grid_size

    # create new img with output_size, black bg
    new_img = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    for index, img in enumerate(images):
        # calc img aspect ratio
        img_ratio = img.shape[1] / img.shape[0]
        # calc target aspect ratio per cell
        target_ratio = cell_width / cell_height

        # resize img to fit in cell
        if img_ratio > target_ratio:
            new_width = cell_width
            new_height = int(cell_width / img_ratio)
        else:
            new_width = int(cell_height * img_ratio)
            new_height = cell_height

        # resize img using inter_lanczos
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        row = index // grid_size
        col = index % grid_size

        # calc x, y offsets for img positioning
        x_offset = col * (cell_width + dist_images) + (cell_width - new_width) // 2
        y_offset = row * (cell_height + dist_images) + (cell_height - new_height) // 2

        # paste resized img in calc pos
        new_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img

    return new_img

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def t5_tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    # Since there's no bos_token_id, simply concatenate the tokenized prompt_chunks with the image_token_index
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def load_pretrained_model(model_cls,
                          model_args,
                          model_path=None,
                          tokenizer_path=None,
                          model_max_length=None,
                          padding_side=None,
                          image_aspect_ratio='pad', # or 'square'
                          mmprojector_repo=None,
                          mmprojector_name=None,
                          device='cuda',
                          cache_dir=HF_CACHE_DIR):
    tokenizer_dict = {}
    if model_max_length:
        tokenizer_dict['model_max_length'] = model_max_length
    if padding_side:
        tokenizer_dict['padding_side'] = padding_side
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, **tokenizer_dict)
    # tokenizer.pad_token = tokenizer.unk_token # could be redundant

    model = model_cls.from_pretrained(model_path, cache_dir=cache_dir)
    
    if mmprojector_repo:
        from huggingface_hub import hf_hub_download
        model_base_name = mmprojector_repo.split('/')[-1]
        
        if cache_dir is not None:
            local_dir = os.path.join(cache_dir, model_base_name)
        elif os.environ.get('HF_HOME') is not None:
            local_dir = os.path.join(os.environ.get('HF_HOME'), model_base_name)
        else:
            local_dir = os.path.join(os.path.expanduser("~"), model_base_name)
        print(f"Downloading projector weights to {local_dir}")
        hf_hub_download(
            repo_id=mmprojector_repo,
            filename=mmprojector_name,
            local_dir=local_dir,
        )
        pretrain_mm_mlp_adapter = os.path.join(local_dir, mmprojector_name)
        model_args.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter # important to set to correct path
        
        model.get_model().initialize_vision_modules(model_args) # This will load the CLIP vision encoder and MLP projector
    else:
        model.resize_token_embeddings(len(tokenizer)) # perhaps not needed

    if not model.get_vision_tower().is_loaded:
        model.get_vision_tower().load_model()
    model.to(device=device, dtype=torch.bfloat16)
    image_processor = model.get_vision_tower().image_processor

    model.requires_grad_(False)
    
    
    # below might be redundant
    model.config.image_aspect_ratio = image_aspect_ratio
    model.config.use_cache = False
    model.config.image_grid_pinpoints = None
    model.config.freeze_mm_mlp_adapter = True

    model = model.eval()
    return tokenizer, model, image_processor