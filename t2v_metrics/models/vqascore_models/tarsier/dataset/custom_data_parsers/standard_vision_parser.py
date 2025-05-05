from typing import Dict, List
from PIL import Image
import random

from .utils import sample_video, read_image, adjust_bbox, filter_ocr_polygon


class VisionParser:
    def __init__(
        self,
        n_frames=8,
        max_n_frames=256,
        is_training=True,
        video_sampling_strategy={},
    ):
        self.n_frames = n_frames
        self.max_n_frames = max_n_frames
        self.is_training = is_training
        self.video_sampling_strategy = video_sampling_strategy

        # fmt: off
        self.data_temp = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image and the video."},
                        # 支持的 image 格式：
                        {"type": "image", "image": {"image_file": "/path/to/image"}},
                        {"type": "image", "image": {"video_file": "/path/to/video", "frame_indices": 0}},
                        # 支持的 video 格式：
                        {"type": "video", "video": {"video_file": "/path/to/video"}},
                        {"type": "video", "video": {"video_file": "/path/to/video", "frame_indices": [0, 1, 2]}},
                        {"type": "video", "video": {"video_file": "/path/to/video", "start_frame": 0, "end_frame": 100}},
                        {"type": "video", "video": {"video_file": "/path/to/video", "time_indices": [0, 1, 2]}},
                        {"type": "video", "video": {"video_file": "/path/to/video", "start_time": 0, "end_time": 100}},
                        {"type": "video", "video": {"image_file": ["/path/to/image"]}, "frame_indices": [0, 1, 2]},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text","text": "xxx"}
                    ]
                }
            ],
            "dataset": "LSMDC",
            "task": "video/caption"
        }
        # fmt: on
    
    def check_format(self, data_dict: Dict, image_processing_config: Dict):
        if image_processing_config.get('do_crop', False) and image_processing_config.get('has_coordinates', False):
            raise ValueError(f'do_crop and has_coordinates cannot be True at the same time!')

    """
    1. 将 messages 中的 image/video 替换成相应的 PIL.Image/List[PIL.Image]
    2. text 的特殊处理：调整 box；过滤面积太小的OCR
    """
    def transform(self, data_dict: Dict, image_processing_config: Dict = None) -> Dict:
        self.check_format(data_dict, image_processing_config)

        self.set_n_frames(data_dict)

        first_image = None # ugly! 需要调整box/过滤面积太小的OCR的数据只有图片任务

        for msg in data_dict['messages']:
            if isinstance(msg['content'], dict):
                msg['content'] = [msg['content']]
            for content in msg['content']:

                if content['type'] == 'image':
                    content['image'] = self.load_image_item(content['image'])
                    if first_image is None:
                        first_image = content['image']
                elif content['type'] == 'video':
                    video = self.load_video_item(content['video'])
                    content['video'] = video.pop('frames')
                    if video:
                        data_dict['extra_info']['frame_disturb_info'] = video.pop('video_info', {})
                elif content['type'] == 'text':
                    pass
                else:
                    raise ValueError(f"content['type']={content['type']} MUST be one of ['image', 'video', 'text']")
        for msg in data_dict['messages']:
            for content in msg['content']:
                if content['type'] == 'text':
                    self.postprocess_text(content, data_dict, image_processing_config, first_image)

        return data_dict['messages']
                
    # set n_frames for each vision item.
    def set_n_frames(self, data_dict):

        if isinstance(self.n_frames, int):
            n_frames = self.n_frames
        else:
            n_frames = random.choice(self.n_frames)
        
        assert n_frames <= self.max_n_frames

        curr_n_frames = 0
        has_dynamic = False
        for msg in data_dict['messages']:
            if isinstance(msg['content'], dict):
                msg['content'] = [msg['content']]

            for content in msg['content']:

                if content['type'] == 'image':
                    curr_n_frames += 1 
                elif content['type'] == 'video':
                    if 'frame_indices' in content['video']:                        
                        curr_n_frames += len(content['video']['frame_indices'])
                        content['video']['n_frames'] = len(content['video']['frame_indices'])
                    elif 'time_indices' in content['video']:
                        curr_n_frames += len(content['video']['time_indices'])
                        content['video']['n_frames'] = len(content['video']['time_indices'])
                    elif 'min_n_frames' in content['video']:
                        content['video']['min_n_frames'] = int(content['video']['min_n_frames'])
                        curr_n_frames += content['video']['min_n_frames']
                        content['video']['n_frames'] = content['video']['min_n_frames']
                        has_dynamic = True        
                    elif 'fps' in content['video']:
                        content['video']['n_frames'] = self.max_n_frames
                        curr_n_frames += self.max_n_frames
                        has_dynamic = True        
                    else:
                        content['video']['n_frames'] = 0
                        has_dynamic = True

        while curr_n_frames < n_frames and has_dynamic:
            for msg in data_dict['messages']:
                for content in msg['content']:
                    if content['type'] == 'video':
                        if 'frame_indices' in content['video']:
                            pass
                        elif 'time_indices' in content['video']:
                            pass
                        else:
                            if curr_n_frames < n_frames:
                                content['video']['n_frames'] += 1
                            curr_n_frames += 1
        
        while curr_n_frames > self.max_n_frames and has_dynamic:
            for msg in data_dict['messages']:
                for content in msg['content']:
                    if content['type'] == 'video':
                        if 'frame_indices' in content['video']:
                            pass
                        elif 'time_indices' in content['video']:
                            pass
                        else:
                            if curr_n_frames > self.max_n_frames:
                                content['video']['n_frames'] -= 1
                            curr_n_frames -= 1
    

        for msg in data_dict['messages']:
            for content in msg['content']:
                if content['type'] == 'video':
                    if 'frame_indices' in content['video']:
                        pass
                    elif 'time_indices' in content['video']:
                        pass
                    else:
                        n = self.video_sampling_strategy.get('force_frames_n_divisible', 1)
                        if n > 1 and content['video']['n_frames'] % n != 0:
                            content['video']['n_frames'] += n - content['video']['n_frames'] % n

    def load_image_item(self, image_item) -> Image.Image:
        """
        image_item:
        {"image_file": {"lq": "/path/to/image"}}
        {"video_file": {"lq": "/path/to/video"}, "frame_indices": 0}
        """

        # check format
        if ("image_file" not in image_item) and ("video_file" not in image_item):
            raise KeyError(f"Key 'image_file' or 'video_file' not found in image_item")
        if 'image_file' in image_item:
            if not isinstance(image_item['image_file'], str):
                raise ValueError(f"{image_item['image_file']} is not a str!")
        if 'video_file' in image_item:
            if not isinstance(image_item['frame_indices'], int):
                raise ValueError(f"{image_item['frame_indices']} is not a int!")

        if 'image_file' in image_item:
            image = read_image(image_item['image_file'])
        else:
            frame_indices = [image_item['frame_indices']]
            image = sample_video(image_item['video_file'], frame_indices = frame_indices)[0]

        return image

    def load_video_item(self, video_item) -> List[Image.Image]:
        """
        video_item:
        {"video_file": {"lq": "/path/to/video"}, "n_frames": 8} 
        {"video_file": {"lq": "/path/to/video"}, "frame_indices": [0, 1, 2], "n_frames": 3} 
        {"video_file": {"lq": "/path/to/video"}, "start_frame": 0, "end_frame": 100, "n_frames": 8}
        {"video_file": {"lq": "/path/to/video"}, "time_indices": [0, 1, 2], "n_frames": 3}
        {"video_file": {"lq": "/path/to/video"}, "start_time": 0, "end_time": 100, "n_frames": 8}
        {"image_file": {"lq": ["/path/to/image"]}, "frame_indices": [0, 1, 2], "n_frames": 3}
        """

        # check format
        if ("image_file" not in video_item) and ("video_file" not in video_item):
            raise KeyError(f"Key 'image_file' or 'video_file' not found in video_item")
    
        video_path = video_item.get('video_file', video_item.get('image_file'))
        n_frames = video_item.get('n_frames', None)
        frame_indices = video_item.get('frame_indices', None)
        start_frame = video_item.get('start_frame', None)
        end_frame = video_item.get('end_frame', None)
        time_indices = video_item.get('time_indices', None)
        start_time = video_item.get('start_time', None)
        end_time = video_item.get('end_time', None)
        mask_boxes = video_item.get('mask_boxes', None)
        fps = video_item.get('fps', None)

        frames, frame_indices = sample_video(
            video_path=video_path,
            frame_indices=frame_indices,
            start_frame=start_frame,
            end_frame=end_frame,
            n_frames=n_frames,
            time_indices=time_indices,
            start_time=start_time,
            end_time=end_time,
            sampling_fps=fps,
            mask_boxes=mask_boxes,
            is_training=self.is_training,
            video_sampling_strategy=self.video_sampling_strategy,
            return_frame_ids=True,
        )

        if self.video_sampling_strategy.get('use_multi_images_for_video', False):
            new_frames = []
            for f in frames:
                new_frames.extend([f, f])
            frames = new_frames

        if isinstance(frame_indices, dict):
            return {
                'frames': frames,
                'video_info': frame_indices
            }
        return {'frames': frames}
    
    def postprocess_text(self, content, data_dict, image_processing_config, first_image):
        if image_processing_config.get('has_coordinates') and image_processing_config.get('do_padding'):
            content['text'] = adjust_bbox(content['text'], frame=first_image)
        if data_dict.get('task') == 'image/OCR' and image_processing_config.get('has_coordinates'):
            content['text'] = filter_ocr_polygon(content['text'])
