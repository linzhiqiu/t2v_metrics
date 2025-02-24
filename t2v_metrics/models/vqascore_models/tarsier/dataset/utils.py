# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
import os
from PIL import Image, ImageSequence
import decord

VALID_DATA_FORMAT_STRING = "Input data must be {'.jpg', '.jpeg', '.png', '.tif'} for image; or {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv', '.gif'}  for videos!"

# 均匀抽帧，必采样首尾帧。
def sample_frame_indices(start_frame, total_frames: int, n_frames: int):
    if n_frames == 1:
        return [0]  # sample first frame in default
    sample_ids = [round(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]
    sample_ids = [i + start_frame for i in sample_ids]
    return sample_ids

def sample_video(
    video_path: str, 
    n_frames: int = None,
    start_time: int = 0,
    end_time: int = -1
    ) -> List[Image.Image]:

    assert os.path.exists(video_path), f"File not found: {video_path}"
    vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
    vr.seek(0)
    total_frames = len(vr)
    fps = vr.get_avg_fps()

    start_frame = 0
    end_frame = total_frames - 1
    if start_time > 0:
        start_frame = min((total_frames-1), int(fps*start_time))
    if end_time > 0:
        end_frame = max(start_frame, int(fps*end_time))
        end_frame = min(end_frame, (total_frames-1))
    frame_indices = sample_frame_indices(
        start_frame=start_frame,
        total_frames=end_frame - start_frame + 1,
        n_frames=n_frames,
    )

    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(f).convert('RGB') for f in frames]
    return frames

def sample_gif(
        gif_path: str,
        n_frames:int = None,
        start_time: int = 0,
        end_time: int = -1
    ) -> List[Image.Image]:

    assert os.path.exists(gif_path), f"File not found: {gif_path}"
    
    gif_frames = Image.open(gif_path)

    start_frame = 0
    end_frame = gif_frames.n_frames - 1
    frame_indices = sample_frame_indices(
        start_frame=start_frame,
        total_frames=end_frame - start_frame + 1,
        n_frames=n_frames,
    )
        
    frames = []
    i = 0
    for frame in ImageSequence.Iterator(gif_frames):
        if i in frame_indices:
            frames.append(frame.convert('RGB'))
        i += 1
    return frames

def sample_image(
    image_path: str, 
    n_frames: int = None,
    start_time: int = 0,
    end_time: int = -1
    ):
    assert os.path.exists(image_path), f"File not found: {image_path}"
    image = Image.open(image_path).convert('RGB')
    return [image]

def get_visual_type(input_file):
    ext = os.path.splitext(input_file)[-1]
    if ext in {'.gif'}:
        return 'gif'
    elif ext in {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv'}:
        return 'video'
    elif ext in {'.jpg', '.jpeg', '.png', '.tif'}:
        return 'image'
    else:
        print(f"{VALID_DATA_FORMAT_STRING} But found {ext}!")
        return 'unk'

def get_benchmarks(benchmarks):
    final_benchmarks = []
    type2bm = {
        'dream': ['dream'],
        'caption': ['msvd-caption', 'msr-vtt-caption', 'vatex-caption'],
        'mc_qa': ['next-qa', 'egoschema', 'mvbench', 'video-mme'],
        'oe_qa': ['msvd-qa', 'msr-vtt-qa', 'tgif-qa', 'anet-qa'],
    }
    for bm in benchmarks:
        bm = bm.lower()
        if bm in final_benchmarks:
            continue
        if bm == 'all':
            for v in type2bm.values():
                final_benchmarks.extend(v)
            return final_benchmarks
        if bm in type2bm:
            final_benchmarks.extend(type2bm[bm])
        else:
            final_benchmarks.append(bm)
    return final_benchmarks

def check_data_format(data):
    for msg in data['messages']:
        if isinstance(msg['content'], dict):
            msg['content'] = [msg['content']]
        for content in msg['content']:
            assert content['type'] in {'image', 'video', 'text'}, f"content['type']={content['type']} MUST be one of ['image', 'video', 'text']"
            if content['type'] != "text":
                media_path_key = f"{content['type']}_file"
                meida_paths = content[content['type']][media_path_key]
                if isinstance(meida_paths, str):
                    meida_paths = [meida_paths]
                for path in meida_paths:
                    assert os.path.exists(path), f"File not found: {path}"

def format_one_sample(media_file=None, prompt="Describe the video in detail."):
    sample = {
        "messages": []
    }
    user_content = {
        "role": "user",
        "content": []
    }
    if media_file is not None:
        media_type = get_visual_type(media_file)
        if media_type in ("video", "gif"):
            media_type = "video"
        media_path_key = f"{media_type}_file"
        user_content["content"].append({
            "type": media_type,
            media_type: {
                media_path_key: media_file,
            }
        })
    user_content["content"].append({
        "type": "text",
        "text": prompt
    })

    assistant_content = {
        "role": "assistant",
        "content": []
    }

    sample["messages"].append(user_content)
    sample["messages"].append(assistant_content)
    if media_file is not None:
        sample["task"] = f"{media_type}/QA"
    else:
        sample["task"] = 'text-only'
    check_data_format(sample)
    return sample


class DictToObject(object):
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
