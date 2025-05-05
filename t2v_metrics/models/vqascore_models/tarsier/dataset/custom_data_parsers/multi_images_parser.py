from typing import Dict, List
import random
import re
from PIL import Image

from .utils import sample_video, read_image

class MultiImagesParser:
    def __init__(
        self,
        n_frames=8,
        is_training=True,
    ):
        self.n_frames = n_frames
        self.is_training = is_training
        # fmt: off
        self.data_temp = {
            "text": [
                [{
                    "prompt": "Describe the image in short.",
                    "response": "A rollerblader rides high in a full pipe while others watch"
                }],
                [{
                    "prompt": "Describe the image in short.",
                    "response": "A woman in winter clothes is on the sidewalk with a phone."
                }]
            ],
            "image": [
                {
                    "image_file": "/mnt/bn/videonaslq/images/flickr30k/images/3371533654.jpg"
                },
                {
                    "image_file": "/mnt/bn/videonaslq/images/coco/train2014/COCO_train2014_000000177950.jpg"
                },
                {
                    "video_file": "/mnt/bn/llmdatalq/jiangnan/video_generation/webvid_10M_download/20230609/videos/011851_011900/1047443473.mp4",
                    "frame_indices": [0, 85, 171, 256, 342, 427, 513, 598]
                }
            ],
            "dataset": "coco",
            "task": "multi_images",
            "image_processing_config": {},
        }
        # fmt: on
    
    def check_format(self, data_dict: Dict, image_processing_config: Dict):
        assert data_dict['dataset'] in ['coco', 'sharegpt4v_cap100k', 'sharegpt4v_mix665k', 'webvid', 'movie'], data_dict

        # 目前多图数据应该没有包含坐标的数据吧
        if image_processing_config.get('has_coordinates', False):
            raise ValueError(f'do_crop and has_coordinates cannot be True at the same time in MultiImagesParser!')
        
        # 检查是否能匹配到坐标
        texts = data_dict['text']
        for text in texts:
            match = re.search(r'\[(\d+(\.\d+)?,\s*)+\d+(\.\d+)?\]', text['prompt'] + text['response'])
            if match:
                print(f'[Warning] 疑似检测到包含坐标的数据：{data_dict}')

    
    def transform(self, data_dict: Dict, image_processing_config: Dict = None) -> Dict:
        self.check_format(data_dict, image_processing_config)

        # shuffle
        texts = data_dict['text']
        images = data_dict['image']
        images = self.load_images(images)
        idxs = list(range(len(texts)))
        random.shuffle(idxs)
        texts = [texts[i] for i in idxs]
        images = [images[i] for i in idxs]

        # sample n_frames
        if isinstance(self.n_frames, int):
            n_frames = random.choice(list(range(1, self.n_frames + 1)))
        else:
            n_frames = random.choice(self.n_frames)
        texts = texts[: n_frames]
        images = images[: n_frames]

        dataset = data_dict['dataset']
        if dataset in ['coco', 'sharegpt4v_cap100k', 'webvid', 'movie']:
            prompt, response = self.transform_for_caption_task(texts, dataset, images)
        else:
            prompt, response = self.transform_for_qa_task(texts, dataset, images)

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
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
    
    def transform_for_caption_task(self, texts, dataset, images):
        idx = random.choice(list(range(len(texts))))

        if dataset == 'coco':
            if len(texts) == 1:
                prompt = 'Describe the image in short.'
            else:
                prompt = f'Describe the images starting from frame {idx + 1} in short in order.'
        elif dataset == 'sharegpt4v_cap100k':
            if len(texts) == 1:
                prompt = 'Describe the image in detail.'
            else:
                prompt = f'Describe the images starting from frame {idx + 1} in detail in order.'
        else:
            if len(texts) == 1:
                prompt = 'Describe the image.'
            else:
                prompt = f'Describe the images starting from frame {idx + 1} in order.'
        response = ''
        for i, text in enumerate(texts):
            if i < idx:
                continue
            if not isinstance(text, dict):
                text = random.choice(text)
            resp = text['response']
            response += f'{resp}\n'
        return prompt, response
    
    def transform_for_qa_task(self, texts, dataset, images):
        prompt, response = '', ''
        for i, text in enumerate(texts):
            if not isinstance(text, dict):
                text = random.choice(text)
            if len(texts) > 1:
                prompt += f'Question for frame {i+1}:\n' + text['prompt'] + '\n'
                response += f'Answer to question of frame {i+1}:\n' + text['response'] + '\n'
            else:
                prompt += text['prompt'] + '\n'
                response += text['response'] + '\n'
        return prompt, response


    def load_images(self, image_items: List[Dict]) -> List[Image.Image]:
        """
        image_items: List[Dict]. each item like:
            {"video_file": "path/to/video", "frame_indices": [1]}
            or
            {"image_file": "path/to/image"}    
        """
        if image_items is None:
            raise ValueError(f'image_items is None!')

        if isinstance(image_items, dict):
            image_items = [image_items]

        images = []

        for image_item in image_items:

            if 'video_file' in image_item:
                file_key = 'video_file'
            elif 'image_file' in image_item:
                file_key = 'image_file'
            else:
                raise KeyError(f'video_file or image_file not in {image_item}')

            file_path = image_item[file_key]
            if file_key == 'video_file':
                frame_indices = image_item.get('frame_indices', None)
                if frame_indices is None:
                    raise ValueError(f'read 0 frame: {image_item}')
                if isinstance(frame_indices, int):
                    frame_indices = [frame_indices]
                frames = sample_video(file_path, frame_indices = frame_indices)
                images.extend(frames)
            else:
                if isinstance(file_path, str):
                    file_path = [file_path]
                images.extend([read_image(f) for f in file_path])

        return images

if __name__ == '__main__':
    # python3 -m xenon_generation.data.custom_data_parsers.multi_images_parser

    from tqdm import tqdm
    from tools.rw_utils import read_jsonlines

    lines = read_jsonlines('/mnt/bn/videonaslq/VideoCaption/datasets_1009/sharegpt4v_cap100k/part_36.jsonl')
    lines = lines[:10]
    parser = MultiImagesParser(n_frames=8)
    for i, l in tqdm(enumerate(lines)):
        l_image_processing_config = l.get('image_processing_config', {})
        messages = parser.transform(l, l_image_processing_config)
        print(messages)