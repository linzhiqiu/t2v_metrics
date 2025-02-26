# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
from io import BytesIO
import base64
import numpy as np
import os, math, cv2, re

import torch
from transformers import StoppingCriteria

import tempfile
from io import BytesIO
from decord import VideoReader, cpu


def read_video_cv2(video_path, all_indices):
    vidcap = cv2.VideoCapture(video_path)
    frames_dict = {}
    max_index = max(all_indices)  # Find the maximum index to avoid unnecessary reading
    count = 0
    success = True
    while success and count <= max_index:
        success, frame = vidcap.read()
        if success and count in all_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            frames_dict[count] = im_pil
        count += 1
    # Now retrieve frames according to all_indices, allowing duplicates
    images = [frames_dict[idx] for idx in all_indices if idx in frames_dict]
    return np.stack([np.array(img) for img in images])

def read_video_decord(video_file, all_indices):
    vr = VideoReader(video_file, num_threads=1, ctx=cpu(0))
    buffer = vr.get_batch(all_indices)
    if isinstance(buffer, torch.Tensor):
        return buffer
    return torch.tensor(vr.get_batch(all_indices).asnumpy())


def read_video_decord_eval(video_file, all_indices):
    vr = VideoReader(video_file)
    buffer = vr.get_batch(all_indices)
    if isinstance(buffer, torch.Tensor):
        return buffer
    return torch.tensor(vr.get_batch(all_indices).asnumpy())

def load_frames_from_video(video_file, all_indices, video_decode_backend="decord", eval_=True):
    video_ending = os.path.splitext(video_file)[1]
    if video_ending in ['.gif', '.webm'] or video_decode_backend=="opencv":
        buffer = read_video_cv2(video_file, all_indices)
    else:
        # Use decord for other video formats
        if eval_:
            buffer = read_video_decord_eval(video_file, all_indices)
        else:
            buffer = read_video_decord(video_file, all_indices)
    return buffer # (T, H, W, C)

def pad_to_center_square(frames, mean_values):
    """
    Pad the given frame or frames numpy array to square dimensions using the mean values as the padding color.
    Handles both single frames (H, W, C) and batches of frames (N, H, W, C).
    Args:
        frames (np.array): The input frame array of shape (H, W, C) or (N, H, W, C).
        mean_values (tuple): Mean values for each channel, typically derived from dataset normalization parameters.
    Returns:
        np.array: The padded frame array with square dimensions.
    """
    if frames.ndim == 3:  # Single frame
        frames = frames[np.newaxis, :]  # Add a batch dimension
    elif frames.ndim != 4:
        raise ValueError("Input array must be either of shape (H, W, C) or (N, H, W, C)")

    N, height, width, channels = frames.shape
    size = max(width, height)
    background_color = np.array(mean_values, dtype=frames.dtype)
    
    # Create a background array with the size and fill it with the mean values
    padded_frames = np.full((N, size, size, channels), background_color, dtype=frames.dtype)

    # Calculate padding offsets
    top, left = (size - height) // 2, (size - width) // 2

    # Place the original frames in the center of the square canvas
    padded_frames[:, top:top + height, left:left + width, :] = frames
    return padded_frames


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        # result.paste(pil_img, (0, 0))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        # result.paste(pil_img, (0, 0))
        return result


def calculate_sample_indices(clip_duration, frames_per_clip, total_frames, original_fps, video_duration, clip_sampling_ratio=1):
    sample_video_fps = frames_per_clip / clip_duration
    num_clips = math.ceil((video_duration / clip_duration) * clip_sampling_ratio)
    frame_step = original_fps / sample_video_fps
    partition_len = total_frames // num_clips
    all_indices, clip_indices, timestamps = [], [], []
    if frame_step > 0.5:
        frame_step = max(1, int(original_fps / sample_video_fps)) #was int/floor
        clip_len = int(frames_per_clip * frame_step) #was int/floor
        sample_len = min(clip_len, total_frames)
        clip_step = (total_frames - clip_len) // max(1, (num_clips - 1)) if total_frames > clip_len else 0
        for i in range(num_clips):
            if partition_len > clip_len:
                start_idx = (partition_len - clip_len) // 2
                end_idx = start_idx + clip_len
                indices = np.arange(start_idx, end_idx, frame_step)
                indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                indices = indices+ i * partition_len

            else:
                
                indices = np.arange(0, sample_len, frame_step)
                if len(indices) < frames_per_clip:
                    padding = np.full(frames_per_clip - len(indices), sample_len)
                    indices = np.concatenate((indices, padding))
                    
                indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

            # Calculate timestamps
            start_time = (indices[0] / original_fps)
            end_time = (indices[-1] / original_fps)
            timestamps.append((start_time, end_time))

    else:
        ## original video FPS too low, we need to sample the same frame multiple times. 
        ##  Generally should not happen.
        # Calculate the number of times each frame should be sampled
        num_sample = int(np.ceil(1 / frame_step))
    
        # Compute the effective clip length considering the frame step
        clip_len = int(frames_per_clip * frame_step)
    
        # Create an expanded list of indices with each frame repeated num_sample times
        indices = np.repeat(np.arange(clip_len), num_sample)

        # Ensure the clip length does not exceed the total number of frames
        clip_len = min(clip_len, len(indices))
        clip_step = (total_frames - clip_len) // max(1, (num_clips - 1)) if total_frames > clip_len else 0
        
        sample_len = min(clip_len, total_frames)
        if len(indices) < frames_per_clip:
            padding = np.full(frames_per_clip - len(indices), sample_len)
            indices = np.concatenate((indices, padding))
    
        # Distribute the indices into clips
        for i in range(num_clips):
            current_clip_indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
            current_clip_indices = current_clip_indices + i * clip_step

            # Append the current clip indices to the list of all clips
            clip_indices.append(current_clip_indices)
            all_indices.extend(current_clip_indices)

            # Calculate timestamps
            start_time = (current_clip_indices[0] / original_fps)
            end_time = (current_clip_indices[-1] / original_fps)
            timestamps.append((start_time, end_time))

    return clip_indices, all_indices, timestamps

def calculate_sample_indices_uniform(frames_per_clip, total_frames, uniform_frame_count, original_fps):

    # Generate indices
    if total_frames >= N:
        # Sample N frames uniformly without replacement
        indices = np.linspace(0, total_frames - 1, N, dtype=int)
    else:
        # Not enough frames; repeat frames to reach N frames
        repeats = math.ceil(N / total_frames)
        base_indices = np.arange(total_frames)
        indices = np.tile(base_indices, repeats)[:N]

    # Split indices into clips
    clip_indices = [
        indices[i * frames_per_clip: (i + 1) * frames_per_clip]
        for i in range(num_clips)
    ]

    # Calculate timestamps for each clip
    timestamps = []
    for clip in clip_indices:
        start_time = clip[0] / original_fps
        end_time = clip[-1] / original_fps
        timestamps.append((start_time, end_time))

    all_indices = indices.tolist()
    return clip_indices, all_indices, timestamps


def get_video_details(fname):
    """ Load video content using Decord """
    assert os.path.exists(fname), f'video path not found {fname}'
    _fsize = os.path.getsize(fname)
    assert _fsize >= 1 * 1024, f"video too short {fname}"
    vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))            
    # Get the total number of frames and the original fps of the video
    total_frames = len(vr)
    original_fps = vr.get_avg_fps()
    video_duration = total_frames / original_fps
    return total_frames, original_fps, video_duration


def get_video_details_cv2(fname):
    """
    Load video content using OpenCV (cv2) and retrieve video details.
    Args:
        fname (str): Path to the video file.
    Returns:
        tuple: A tuple containing:
            - total_frames (int): Total number of frames in the video.
            - original_fps (float): Frames per second of the video.
            - video_duration (float): Duration of the video in seconds.
    Raises:
        AssertionError: If the file does not exist or is too short.
        ValueError: If the video cannot be opened or FPS is zero.
    """
    # Check if the file exists
    assert os.path.exists(fname), f'Video path not found: {fname}'
    
    # Check if the file size is at least 1 KB
    _fsize = os.path.getsize(fname)
    assert _fsize >= 1 * 1024, f"Video too short: {fname}"
    
    # Open the video file
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {fname}")
    
    # Retrieve the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Retrieve the frames per second (FPS)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        cap.release()
        raise ValueError(f"Failed to get FPS for video file: {fname}")
    
    # Calculate the video duration in seconds
    video_duration = total_frames / original_fps
    
    # Release the video capture object
    cap.release()
    
    return total_frames, original_fps, video_duration


def split_into_clips(video, frames_per_clip):
    """ Split video into a list of clips """
    fpc = frames_per_clip
    nc = len(video) // frames_per_clip
    return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

def process_image(vision_processors, frames_per_clip, image):
    mm_data = []
    for vision_processor in vision_processors:
        tmp = expand2square(image, tuple(int(x * 255) for x in vision_processor.image_mean))
        tmp = np.expand_dims(np.asarray(tmp), 0)
        tmp = vision_processor.preprocess(tmp, return_tensors='pt')['pixel_values'][0].unsqueeze(0)
        if len(tmp.shape)==4:
            ## image, need B, T, C, W, H
            tmp = tmp.unsqueeze(1)
            tmp = tmp.repeat_interleave(frames_per_clip, dim=1)
        else:
            ## video, need B, C, T, W, H
            if tmp.shape[1]==1:
                tmp = tmp.repeat_interleave(frames_per_clip, dim=1)
            else:
                tmp = tmp.repeat_interleave(frames_per_clip, dim=2)
            
        mm_data.append(tmp)
    return mm_data

def process_video(vision_processors, frames_per_clip, buffer):
    mm_data=[]
    for vision_processor in vision_processors:
        centered_buffer = pad_to_center_square(buffer, tuple(int(x * 255) for x in vision_processor.image_mean))
        processed_clips = []
        for clip in split_into_clips(centered_buffer, frames_per_clip):
            clip = vision_processor.preprocess(clip, return_tensors='pt')['pixel_values']
            if type(clip) is list:
                assert len(clip)==1, "LazyVideoDataset: error, vision processor returned clip that is list of len>1 ."
                clip = clip[0]
            processed_clips.append(clip)
        mm_data.append(torch.stack(processed_clips))
    return mm_data

def load_video(video_file, vision_processors, clip_duration, frames_per_clip, clip_sampling_ratio=1, video_decode_backend='decord', eval_=False):
    total_frames, original_fps, video_duration = get_video_details(video_file)
    _, all_indices, timestamps = calculate_sample_indices(clip_duration, frames_per_clip, total_frames, original_fps, video_duration, clip_sampling_ratio=clip_sampling_ratio)
    buffer = load_frames_from_video(video_file, all_indices, video_decode_backend, eval_)
    mm_data = process_video(vision_processors, frames_per_clip, buffer)
    return mm_data, timestamps

def load_video_uniform(video_file, vision_processors, clip_duration, frames_per_clip, clip_sampling_ratio=1, video_decode_backend='decord', eval_=False, uniform_sampling=8):
    total_frames, original_fps, video_duration = get_video_details(video_file)
    all_indices = np.linspace(0, total_frames-1, uniform_sampling, dtype=int)
    print('using uniform frame sampled, sampled: ', len(all_indices), ' frames')
    buffer = load_frames_from_video(video_file, all_indices, video_decode_backend, eval_)
    mm_data = process_video(vision_processors, frames_per_clip, buffer)
    return mm_data, []


class ApolloMMLoader:
    def __init__(self, vision_processors, clip_duration, frames_per_clip, num_repeat_token, device, model_max_length = 32768, clip_sampling_ratio=1, video_decode_backend="decord"):
        self.vision_processors=vision_processors
        self.clip_duration=clip_duration
        self.device=device
        self.frames_per_clip=frames_per_clip
        self.num_repeat_token = num_repeat_token
        self.clip_sampling_ratio=clip_sampling_ratio
        self.model_max_length=model_max_length
        self.video_decode_backend=video_decode_backend
        # self.vidprompt = lambda num_clips, video_duration : f"You are provided the following series of {num2words(num_clips)}, {self.clip_duration} second clips from a {datetime.timedelta(seconds=video_duration)} [H:MM:SS] video.\n"
    
    def load_video(self, video_file):
        total_frames, original_fps, video_duration = get_video_details(video_file)
        
        total_token_at_full_coverage = video_duration  * self.num_repeat_token / self.clip_duration
        clip_sampling_ratio = min(1, (self.model_max_length * self.clip_sampling_ratio) / total_token_at_full_coverage)
        _, all_indices, timestamps = calculate_sample_indices(self.clip_duration, self.frames_per_clip, total_frames, original_fps, video_duration, clip_sampling_ratio=clip_sampling_ratio)
        video, timestamps = load_video(video_file, self.vision_processors, self.clip_duration, self.frames_per_clip, clip_sampling_ratio=clip_sampling_ratio, eval_=True)
        
        num_clips =  len(video[0]) 
        # num_tokens = num_clips * self.num_repeat_token
        video = [v.to(device=self.device, dtype=torch.bfloat16) for v in video]
        # replace_string = self.vidprompt(num_clips, video_duration)

        # temporal_prompt = [f"{round(clip[0], 1)}-{round(clip[1], 1)} seconds: {X_TOKEN['video'] * self.num_repeat_token}" for clip in timestamps]
        # temporal_prompt = ',\n'.join(temporal_prompt)
        # replace_string = replace_string + temporal_prompt
        
        # return video, replace_string
        return video
    
    def load_image(self, image_file):
        print('implement image loading')
        return None


def get_frame_from_vcap(vidcap, num_frames=10, fps=None, frame_count=None):
    import cv2

    if fps == None or frame_count == None:
        # if one of fps or frame_count is None, still recompute
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or frame_count == 0:
        print("Video file not found. return empty images.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames
    
    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    if frame_interval == 0 and frame_count <= 1:
        print("frame_interval is equal to 0. return empty image.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames
    # print("duration:", duration, "frames:", frame_count, "intervals:", frame_interval)

    images = []
    count = 0
    success = True
    frame_indices = np.linspace(0, frame_count - 2, num_frames, dtype=int)

    while success:
        # print("frame_count:", frame_count, "count:", count, "num_frames:", num_frames, "frame_interval:", frame_interval)
        if frame_count >= num_frames:
            success, frame = vidcap.read()
            if count in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                if len(images) >= num_frames:
                    return images
            count += 1
        else:
            # Left padding frames if the video is not long enough
            success, frame = vidcap.read()
            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                count += 1
            elif count >= 1:
                width, height = images[-1].size
                images = [Image.new("RGB", (width, height))] * (num_frames - len(images)) + images
                print("padding frames:", (num_frames - len(images)))
                return images
            else: 
                break
    raise ValueError("Did not find enough frames in the video. return empty image.")


def opencv_extract_frames(vpath_or_bytesio, frames=6, fps=None, frame_count=None):
    """
    Extract frames from a video using OpenCV.
    Args:
        vpath_or_bytesio (str or BytesIO): Path to the video file or BytesIO object containing the video.
        frames (int): Number of frames to extract from the video.
    Returns:
        list: List of PIL Images extracted from the video.
    Raises:
        NotImplementedError: If the type of `vpath_or_bytesio` is not supported.
    """
    import cv2

    if isinstance(vpath_or_bytesio, str):
        vidcap = cv2.VideoCapture(vpath_or_bytesio)
        return get_frame_from_vcap(vidcap, frames, fps=fps, frame_count=frame_count)
    elif isinstance(vpath_or_bytesio, (BytesIO,)):
        # assuming mp4
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(vpath_or_bytesio.read())
            temp_video_name = temp_video.name
            vidcap = cv2.VideoCapture(temp_video_name)
            return get_frame_from_vcap(vidcap, frames, fps=fps, frame_count=frame_count)
    else:
        raise NotImplementedError(type(vpath_or_bytesio))
