from typing import Dict, List
import random
from PIL import Image, ImageDraw, ImageFont

from .utils import sample_video


class VideoPermutationParser:
    def __init__(
        self,
        n_frames=8,
        is_training=True,
        frame_nums = list(range(8, 25)),
        video_sampling_strategy={},
    ):
        self.n_frames = n_frames
        self.is_training = is_training
        self.frame_nums = frame_nums
        self.video_sampling_strategy = video_sampling_strategy
        # fmt: off
        self.data_temp = {
            "text": [{
                "prompt": "<video>",
                "response": ""
            }],
            "video": [{
                "video_file": {
                    "yg": "/mnt/bn/videonasyg/videos/webvid_10M_download/011851_011900/1047443473.mp4",
                    "lq": "/mnt/bn/llmdatalq/jiangnan/video_generation/webvid_10M_download/20230609/videos/011851_011900/1047443473.mp4"
                },
                "frame_indices": [0, 85, 171, 256, 342, 427, 513, 598]
            }],
        }
        # fmt: on

    def check_format(self, data_dict: Dict):
        pass
        # for k in self.data_temp.keys():
        #     assert k in data_dict

    def transform(self, data_dict: Dict, image_processing_config: Dict = None) -> Dict:
        self.check_format(data_dict)

        frames = self.load_video_item(data_dict['video'][0])

        # frames = self.add_text_to_frames(frames) # for debug

        idxs = list(range(1, len(frames) + 1))
        random.shuffle(idxs)

        prefix_len = int(3/8*len(idxs))

        shuffled_frames = [frames[i-1] for i in idxs]

        prompt = f'Output the correct chronological order of scrambled video frames. The order of the first {prefix_len} ones are:\n'
        prompt += '\n'.join([str(i) for i in idxs[: prefix_len]]) + '\nOutput the order of the following frames:'
        response = '\n'.join([str(i) for i in idxs[prefix_len: ]])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": shuffled_frames},
                    {"type": "text", "text": prompt},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            }
        ]

        return messages


    def load_video_item(self, video_item) -> List[Image.Image]:
        """
        video_item:
        {"video_file": "/path/to/video", "n_frames": 8} 
        {"video_file": "/path/to/video", "frame_indices": [0, 1, 2], "n_frames": 3} 
        {"video_file": "/path/to/video", "start_frame": 0, "end_frame": 100, "n_frames": 8}
        {"video_file": "/path/to/video", "time_indices": [0, 1, 2], "n_frames": 3}
        {"video_file": "/path/to/video", "start_time": 0, "end_time": 100, "n_frames": 8}
        {"image_file": ["/path/to/image"], "frame_indices": [0, 1, 2], "n_frames": 3}
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

        n_frames = random.choice(self.frame_nums)
        n = self.video_sampling_strategy.get('force_frames_n_divisible', 1)
        if n > 1 and n_frames % n != 0:
            n_frames += n - n_frames % n

        frames, frame_indices = sample_video(
            video_path=video_path,
            frame_indices=frame_indices,
            start_frame=start_frame,
            end_frame=end_frame,
            n_frames=n_frames,
            time_indices=time_indices,
            start_time=start_time,
            end_time=end_time,
            mask_boxes=mask_boxes,
            is_training=self.is_training,
            video_sampling_strategy=self.video_sampling_strategy,
            return_frame_ids=True,
        )
        return frames
    

    def add_text_to_frames(self, frames: List[Image.Image]):
        new_frames = []
        for i, image in enumerate(frames):
            draw = ImageDraw.Draw(image)
            
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 100)
            text_position = (50, 50)
            text_content = f'{i+1}'
            text_color = (255, 0, 0)
            draw.text(text_position, text_content, font=font, fill=text_color)
            new_frames.append(image)
        return new_frames

