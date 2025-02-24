from typing import List, Dict, Union
import os
import random
import tempfile
from PIL import Image, ImageSequence
import base64
import io
import re
import uuid
import json
import numpy as np
import pyarrow.fs as pf
import func_timeout
from func_timeout import func_set_timeout
import math

# fmt: on
import decord
# fmt: off


def denorm_box(points, height, width):
    new_points = []
    for p in points:
        new_points.append((round(p[0] * width), round(p[1] * height)))
    return new_points

def process_image_for_tiktok(frames: List[Image.Image], mask_boxes):
    mask_boxes = mask_boxes[:len(frames)]
    frames = [np.array(f) for f in frames]
    # assert len(mask_boxes) == len(frames)
    height, width = frames[0].shape[:2]

    new_frames = []
    for boxes, frame in zip(mask_boxes, frames):
        left, top, right, bottom = 0, 0, width, height
        for box in boxes:
            pts = np.array(denorm_box(box, height, width), np.int32)
            upper_bound = max([p[1] for p in pts]) + 30
            if bottom > upper_bound:
                bottom = upper_bound
            frame[pts[0][1]: pts[2][1], pts[0][0]: pts[1][0]] = 0
        
        new_frames.append(Image.fromarray(frame[top: bottom, left: right]))
    return new_frames

# 先将视频分成 n_frames 份。训练时，每份随机抽一帧；测试时，每份抽中间的那一帧。
def _sample_frame_indices_v2(
        total_frames: int, 
        n_frames: int, 
        is_training=False, 
        video_sampling_strategy = {},
    ):
    total_frames_idxs = list(range(total_frames))
    if total_frames <= n_frames: 
        return total_frames_idxs
    k, m = divmod(total_frames, n_frames)
    frame_splits = [total_frames_idxs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(n_frames))]
    if is_training:
        sample_ids = [random.choice(i) for i in frame_splits]
    else:
        sample_ids = [i[(len(i)+1)//2-1] for i in frame_splits]
    return sample_ids

# 均匀抽帧，必采样首尾帧。
def _sample_frame_indices_v1(total_frames: int, n_frames: int, is_training=False, video_sampling_strategy = {}):
    if n_frames == 1:
        return [0]  # sample first frame in default
    if total_frames <= n_frames:
        return list(range(total_frames))
    sample_ids = [round(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]
    return sample_ids

def conduct_disturb_frame(frame_indices):
    disturb_type = random.choice(['exchange', 'crop', 'reverse', 'discard'])
    n_frames = len(frame_indices)
    frame_indices_new = []
    if disturb_type == 'exchange':
        # 均等分成4个segments, 随机交换两个segment
        seg_len = math.ceil(n_frames / 4)
        seg_idxs = list(range(0, n_frames, seg_len))
        target_idxs = random.sample(range(0, 4), 2)
        seg_idxs[target_idxs[0]], seg_idxs[target_idxs[1]] = seg_idxs[target_idxs[1]], seg_idxs[target_idxs[0]]
        for idx in seg_idxs:
            frame_indices_new += frame_indices[idx: idx+seg_len]
    elif disturb_type == 'crop':
        # 随机截取出3/4时长，再采均匀n_frames帧
        crop_len = math.ceil(n_frames / 4)
        idx_s = random.choice(range(0, crop_len+1))
        idx_e = n_frames - 1 - (crop_len - idx_s)
        frame_indices_new = np.linspace(frame_indices[idx_s], frame_indices[idx_e], n_frames, dtype=int).tolist()
    elif disturb_type == 'reverse':
        # 随机选择长度为[1/2, 1]时长的片段进行顺序颠倒
        reverse_len = math.ceil(random.uniform(0.5,1) * n_frames)
        idx_s = random.choice(range(0, n_frames-reverse_len+1))
        idx_e = idx_s + reverse_len - 1
        frame_indices_new = frame_indices[:idx_s] + list(reversed(frame_indices[idx_s: idx_e+1])) + frame_indices[idx_e+1:]
    elif disturb_type == 'discard':
        # 随机丢弃一半帧
        frame_indices_new = random.sample(frame_indices, n_frames//2)
        frame_indices_new.sort()
    return disturb_type, frame_indices_new

@func_set_timeout(60)
def _download_file(path):
    if path.startswith("hdfs"):
        local_path = os.path.join(tempfile.gettempdir(), f'{uuid.uuid4()}_' + os.path.basename(path))

        fs = pf.HadoopFileSystem.from_uri(uri="hdfs://harunava")
        hdfs_file = fs.open_input_file(path)
        file_size = hdfs_file.size()
        if file_size > 1024 * 1024 * 1024: # 1G
            os.system(f"hadoop fs -get --ct 8 -c 512 '{path}' '{local_path}' > /dev/null 2>&1") 
        elif file_size > 1024 * 1024 * 100: # 100M
            os.system(f"hadoop fs -get '{path}' '{local_path}' > /dev/null 2>&1") 
        else:
            local_fs = pf.LocalFileSystem()
            with local_fs.open_output_stream(local_path) as local_file:
                while True:
                    chunk = hdfs_file.read(1024 * 1024 * 100)  # Reading 1MB chunks, you can adjust this as needed
                    if not chunk:
                        break
                    local_file.write(chunk)
    else:
        local_path = path

    if not os.path.exists(local_path):
        raise FileNotFoundError(f'{local_path}')

    return local_path

def download_file(path):
    try:
        # with timer(f'Download {path}'):
        return _download_file(path)
    except func_timeout.exceptions.FunctionTimedOut as e:
        raise ValueError(e)

class VideoReader:
    def __init__(self, path: str) -> None:
        self.path = path
        self.local_path = self.preprocess()
        self.vr = decord.VideoReader(self.local_path, num_threads=1, ctx=decord.cpu(0), fault_tol=1)
        self.vr.seek(0)
        self._length = len(self.vr)
        self._fps = self.vr.get_avg_fps()
    
    @property
    def length(self):
        return self._length
    
    @property
    def fps(self):
        return self._fps

    def sample(self, frame_indices) -> List[Image.Image]:
        frames = self.vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(f).convert('RGB') for f in frames]
        return frames

    def preprocess(self):
        return download_file(self.path)

    def postprocess(self):
        if self.path.startswith("hdfs"):
            os.remove(self.local_path)

class ImageSeqReader:
    def __init__(self, path: List[str]) -> None:
        self.path = path
        self.local_path = self.preprocess()
        self._length = len(self.local_path)
        self._fps = None
    
    @property
    def length(self):
        return self._length
    
    @property
    def fps(self):
        return self._fps

    def sample(self, frame_indices):
        return [read_image(self.local_path[i]) for i in frame_indices]

    def preprocess(self):
        local_paths = []
        for p in self.path:
             local_paths.append(p)
        return local_paths

    def postprocess(self):
        pass

class GIFReader:
    def __init__(self, path: str) -> None:
        self.path = path
        self.local_path = self.preprocess()
        self.gif = Image.open(self.local_path)
        self._length = self.gif.n_frames
        duration = self.gif.info.get('duration', 0) / 1000  # 转换为秒
        if duration > 0:
            self._fps = 1 / duration
        else:
            self._fps = None
    
    @property
    def length(self):
        return self._length
    
    @property
    def fps(self):
        return self._fps

    def sample(self, frame_indices):
        frames = []
        i = 0
        for frame in ImageSequence.Iterator(self.gif):
            if i in frame_indices:
                frames.append(frame.convert('RGB'))
            i += 1
        return frames

    def preprocess(self):
        return download_file(self.path)

    def postprocess(self):
        if self.path.startswith("hdfs"):
            os.remove(self.local_path)

def check_frame_indices(frame_indices, total_frames, video_path):
    if frame_indices[-1] == total_frames:
        frame_indices[-1] = total_frames - 1

    valid_frame_indices = [i for i in frame_indices if i >= 0 and i < total_frames]

    if len(valid_frame_indices) != len(frame_indices):
        print(f'[Error] frame out of index. video_path={video_path}, frame_indices={frame_indices}, total_frames={total_frames}', flush=True)
    
    return valid_frame_indices


def sample_video(
    video_path: Union[str, List[str]], 
    frame_indices: List[int] = None, 
    start_frame:int=None, 
    end_frame:int=None, 
    n_frames:int = None,
    time_indices: List[float] = None,
    start_time:int=None,
    end_time:int=None,
    sampling_fps:float=None,
    mask_boxes=None,
    is_training:bool=False,
    video_sampling_strategy={'video_sampler_version': 'v1'},
    return_frame_ids: bool=False,
    ) -> List[Image.Image]:

    do_frame_disturb = video_sampling_strategy.get('do_frame_disturb', False)

    if isinstance(video_path, str):
        if video_path.endswith('.gif'):
            reader = GIFReader(video_path)
        else:
            reader = VideoReader(video_path)
    else:
        reader = ImageSeqReader(video_path)
    
    total_frames = reader.length
    fps = reader.fps

    if sampling_fps is not None:
        frame_indices = list(range(0, total_frames, round(fps / sampling_fps)))
        if len(frame_indices) > n_frames:
            frame_indices = None

    if time_indices is not None:
        frame_indices = [round(float(i) * fps) for i in time_indices]

    if start_time is not None and end_time is not None:
        start_frame = round(start_time * fps)
        end_frame = round(end_time * fps)   

    if frame_indices is None:
        start_frame = 0 if start_frame is None else round(start_frame)
        end_frame = total_frames - 1 if end_frame is None else round(end_frame)

        if end_frame == total_frames:
            end_frame -= 1

        if video_sampling_strategy['video_sampler_version'] == 'v1':
            # 均匀抽帧，必采样首尾帧。
            frame_indices = _sample_frame_indices_v1(end_frame - start_frame + 1, n_frames, is_training, video_sampling_strategy)
        elif video_sampling_strategy['video_sampler_version'] == 'v2':
            frame_indices = _sample_frame_indices_v2(end_frame - start_frame + 1, n_frames, is_training, video_sampling_strategy)
        else:
            raise ValueError(f"video_sampler_version={video_sampling_strategy['video_sampler_version']} must be 'v1' or 'v2'")
        frame_indices = [i + start_frame for i in frame_indices]
    
    frame_indices = check_frame_indices(frame_indices, total_frames, video_path)

    if do_frame_disturb:
        frame_disturb_type, frame_indices_new = conduct_disturb_frame(frame_indices)
        frame_indices_raw = frame_indices[:]
        frame_indices = frame_indices_new
        
    frames = reader.sample(frame_indices)
    if mask_boxes is not None:
        frames = process_image_for_tiktok(frames, mask_boxes)
    
    n = video_sampling_strategy.get('force_frames_n_divisible', 1)
    if n > 1 and len(frames) % n != 0:
        new_n = n - len(frames) % n
        frames.extend([Image.new(mode='RGB', size=frames[-1].size) for _ in range(new_n)])
            
    reader.postprocess()

    if do_frame_disturb:
        return frames, {"frame_indices": frame_indices, "disturb_type": frame_disturb_type, "frame_indices_raw": frame_indices_raw}
    if return_frame_ids:
        return frames, frame_indices
    return frames



def load_image_from_base64String(img_path):
    img = base64.b64decode(open(img_path, "rb").read())
    buf = io.BytesIO(img)
    img = Image.open(buf)
    return img

def read_image(image_path):
    local_file = download_file(image_path)

    if local_file.endswith('.dat'):
        image = load_image_from_base64String(local_file)
    else:
        image = Image.open(local_file).convert('RGB')
    if image_path.startswith("hdfs"):
        os.remove(local_file)
    return image


def adjust_bbox(text, frame):
    
    width, height = frame.size
    new_text = []
    start_idx = 0
    for match in re.finditer(r'\[(\d+(\.\d+)?,\s*)+\d+(\.\d+)?\]', text):
        coordinate_matches = re.findall(r"([0-9.]+)", match.group(0))
        xys = [float(coord) for coord in coordinate_matches]

        new_xys = []
        for i in range(len(xys)):
            p = xys[i]

            if width == height:
                pass
            
            if width > height and i % 2 != 0:
                p = xys[i] * height
                p += (width - height) // 2
                p = round(p / width, 2)
            
            if height > width and i % 2 == 0:
                p = xys[i] * width
                p += (height - width) // 2
                p = round(p / height, 2)
            
            new_xys.append(p)
        
        new_text.append(text[start_idx: match.span()[0]])
        new_text.append(str(new_xys))
        start_idx = match.span()[1]
    new_text.append(text[start_idx: ])
    text = ''.join(new_text)
        

    return text

def bbox_area(vertices, convert_format = True):
    if convert_format:
        vertices = list(zip(vertices[::2], vertices[1::2]))
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    return abs((x1 - x0) * (y1 - y0))

def polygon_area(vertices, convert_format = True):
    if convert_format:
        vertices = list(zip(vertices[::2], vertices[1::2]))
    n = len(vertices)  # 多边形顶点的数量
    if n == 2:
        return bbox_area(vertices, convert_format=False)
    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def get_text_len(text_line):
    l = 0
    for c in text_line:
        if '\u4e00' <= c <= '\u9fff':
            l += 1
        else:
            l += 0.5
    return l

def filter_ocr_polygon(response, area_threshold=0.0005):
    try:
        resp = json.loads(response)
    except:
        return response
    new_resp = []
    for coords, text_line in resp:
        area = polygon_area(coords, convert_format=True)
        text_len = get_text_len(text_line)
        if text_len == 0:
            continue
        if area / text_len < area_threshold:
            continue
        new_resp.append([coords, text_line])
    new_resp = json.dumps(new_resp, ensure_ascii=False)
    
    return new_resp

def put_pred_to_data_dict(prediction, data_dict):
    msg = data_dict['messages'][-1]
    if msg['role'] == 'assistant':
        msg['content'][-1]['text'] = prediction
    else:
        data_dict['messages'].append({
            "role": "assistant",
            "content": [{"type": "text", "text": prediction}]
        })

def get_prompt_from_data_dict(data_dict):
    prompt = ""
    for msg in data_dict['messages']:
        role = msg['role']
        assert role in {'system', 'user', 'assistant'}
        for content in msg['content']:
            if content['type'] == 'text':
                if content['text']:
                    prompt += f"[{role}]: {content['text']}"
            elif content['type'] == 'image':
                prompt += f"[{role}]: <image>"
            elif content['type'] == 'video':
                prompt += f"[{role}]: <video>"
            prompt += '\n'
    return prompt
