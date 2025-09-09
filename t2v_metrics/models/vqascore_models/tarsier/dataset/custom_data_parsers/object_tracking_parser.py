from typing import Dict
import random
import re

from torchvision import transforms

from .utils import sample_video

def return_same(x):
    return x

def _bbox_transform_for_padding(bbox, frame):
    w1, h1, w2, h2 = bbox
    width, height = frame.size
    if width == height:
        pass
    elif width > height:
        h1 += (width - height) // 2
        h2 += (width - height) // 2
        height = width
    else:
        w1 += (height - width) // 2
        w2 += (height - width) // 2
        width = height
    new_bbox = [w1 / width, h1 / height, w2 / width, h2 / height]
    new_bbox = [round(i, 2) for i in new_bbox]
    return new_bbox

def _bbox_transform_for_resize(bbox, frame):
    w1, h1, w2, h2 = bbox
    width, height = frame.size
    new_bbox = [w1 / width, h1 / height, w2 / width, h2 / height]
    new_bbox = [round(i, 2) for i in new_bbox]
    return new_bbox

class InAndOutCropAndResize(object):
    """Crop and resize for in_and_out boxes data according to yuchen
    Args:
        size: tuple of (width, height)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        w = img.width
        h = img.height
        x0 = int(w * 0.5 - h * 0.375)
        y0 = int(h * 0.125)
        x1 = int(w * 0.5 + h * 0.375)
        y1 = int(h * 0.875)
        img = img.crop((x0, y0, x1, y1)).resize(self.size)
        return img


class ObjectTrackingParser:
    def __init__(
        self,
        n_frames = 8,
        max_objects = 3,
        is_training=True,
    ):
        self.n_frames = n_frames
        self.max_objects = max_objects
        self.is_training = is_training
        self.img_transform = self.get_img_transform()
        # fmt: off
        self.data_temp = {
            "video_file": "/mnt/bn/llmdatalq/jiaxin/hdvila/20230926/saved/saved_video_clips/0076/lOjn__YCec4.624.1104.mp4",
            "frame_indices": [154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202],
            "objects": {
                "0": {
                    "phrase": "person",
                    "all_frame_bounding_boxes": [[2, 0, 255, 250], [17, 0, 255, 251], [35, 0, 255, 253], [44, 0, 255, 255], [52, 0, 255, 255], [54, 0, 255, 255], [63, 0, 255, 255], [60, 0, 255, 255], [54, 0, 253, 255], [43, 0, 250, 255], [36, 1, 249, 255], [36, 0, 252, 254], [41, 0, 252, 254], [61, 0, 255, 253], [68, 4, 255, 255], [74, 8, 255, 255], [91, 3, 255, 255]]
                }
            },
            "task": "object_tracking",
            "dataset": "hdvila"
        }
        # fmt: on

    def check_format(self, data_dict: Dict, image_processing_config: Dict):
        # box tracking 数据不支持 do_crop！！！
        if image_processing_config.get('do_crop', False):
            raise ValueError(f'do_crop is not supported in ObjectTrackingParser!')

    def transform(self, data_dict: Dict, image_processing_config: Dict = None) -> Dict:
        self.check_format(data_dict, image_processing_config)

        bbox_transform = _bbox_transform_for_padding if image_processing_config['do_padding'] else _bbox_transform_for_resize

        # sample n_frames
        if isinstance(self.n_frames, int):
            n_frames = self.n_frames
        else:
            n_frames = random.choice(self.n_frames)
        total_frames = list(range(len(data_dict['frame_indices'])))
        idxs = random.sample(total_frames, min(n_frames, len(total_frames)))
        idxs.sort()

        frame_indices = [data_dict['frame_indices'][i] for i in idxs]
        frames = sample_video(data_dict['video_file'], frame_indices=frame_indices)
        img_transform = self.img_transform[data_dict['dataset']]
        frames = [img_transform(f) for f in frames]

        objects = []
        for _, o in data_dict['objects'].items():
            if o is None:
                continue
            all_frame_bounding_boxes = [o['all_frame_bounding_boxes'][i] for i in idxs]
            all_frame_bounding_boxes_t = []
            for bbox, frame in zip(all_frame_bounding_boxes, frames):
                all_frame_bounding_boxes_t.append(bbox_transform(bbox, frame))
            objects.append(all_frame_bounding_boxes_t)
            if len(objects) >= self.max_objects:
                break

        prompt = "Given the bounding box coordinates of these objects in the first frame, output the bounding box coordinates in the following frames.\n{}"
        response = ''

        object_info = ''
        for i, o in enumerate(objects):
            object_info += f'object {i+1}: {o[0]}\n'
            response += f'object {i+1}: {o[1:]}\n'
        response = response.strip()
        prompt = prompt.format(object_info)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt}
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

    def get_img_transform(self):
        return {
            'webvid': return_same,
            'hdvila': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=(256, 256))
            ]),
            'hdvila_in_and_out_boxes': InAndOutCropAndResize(size=(256, 256))
        }
