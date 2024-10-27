from abc import abstractmethod
from typing import List, TypedDict, Union, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from .constants import HF_CACHE_DIR
import os

from .models.vqascore_models.mm_utils import *



class ImageTextDict(TypedDict):
    images: List[str]
    texts: List[str]

class Score(nn.Module):

    def __init__(self,
                 model: str,
                 device: str='cuda',
                 cache_dir: str=HF_CACHE_DIR,
                 **kwargs):
        """Initialize the ScoreModel
        """
        super().__init__()
        assert model in self.list_all_models()
        self.device = device
        self.model = self.prepare_scoremodel(model, device, cache_dir, **kwargs)
        self.model_name = model
    
    @abstractmethod
    def prepare_scoremodel(self,
                           model: str,
                           device: str,
                           cache_dir: str,
                           **kwargs):
        """Prepare the ScoreModel
        """
        pass
    
    @abstractmethod
    def list_all_models(self) -> List[str]:
        """List all available models
        """
        pass

    def forward(self,
                videos: Optional[Union[str, List[str]]]=None,
                num_frames: Optional[int]=8,
                concatenate: Optional[str]='horizontal',
                images: Optional[Union[str, List[str]]]=None,
                texts: Optional[Union[str, List[str]]]=None,
                **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s)/video(s) and the text(s)
        If there are m images/videos and n texts, return a m x n tensor
        """
        delete_images=False
        if videos is not None:
            
            if isinstance(videos, str):
                videos = [videos]
            
            assert any(videos[0][-4:] in extension for extension in ['.mp4', '.avi', '.mov', '.mkv']), 'Video file type not supported'
            
            # if any(name in self.model_name.lower() for name in ['clip', 'blip', 'llava-v1.5', 'llava-v1.6', 'hpsv2', 'pickscore', 'imag-reward']):
            if self.model.video_mode == "concat":
                delete_images=True
                processed_images = []
                for video in videos:
                    # Extract frames
                    output_dir = f"temp_{os.path.basename(video)}"
                    extract_frames(video, num_frames, output_dir)
                    
                    # Read extracted frames
                    frame_images = [cv2.imread(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.endswith('.jpg')]
                    
                    # Concatenate frames
                    if concatenate == "horizontal":
                        concat_image = concatenate_images_horizontal(frame_images, dist_images=10)
                    elif concatenate == "vertical":
                        concat_image = concatenate_images_vertical(frame_images, dist_images=10)
                    elif concatenate == "grid":
                        concat_image = concatenate_images_grid(frame_images, dist_images=10, output_size=(1024, 1024))
                    else:
                        raise ValueError("Invalid concatenation method")
                    
                    # Save concatenated image
                    output_path = f"concat_{os.path.basename(video)}.jpg"
                    cv2.imwrite(output_path, concat_image)
                    processed_images.append(output_path)
                    
                    # Clean up temporary directory
                    for f in os.listdir(output_dir):
                        os.remove(os.path.join(output_dir, f))
                    os.rmdir(output_dir)
                    
                images = processed_images
            elif self.model.video_mode == "direct":
                images = videos
            else:
                print(f"Invalid `video_mode` for the given model. Please check model's class attributes")
        elif self.model.allows_image:
        
            if isinstance(images, str):
                images = [images]
            if isinstance(texts, str):
                texts = [texts]
        else:
            print(f'The model does not support image-only inference. Please try again.')
            return
        
        scores = torch.zeros(len(images), len(texts)).to(self.device)
        for i, image in enumerate(images):
            scores[i] = self.model.forward([image] * len(texts), texts, **kwargs)
        
        if delete_images:
            for f in processed_images:
                os.remove(f)
        return scores
    
    def batch_forward(self,
                      dataset: List[ImageTextDict],
                      batch_size: int=16,
                      num_frames: int=4,
                      **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s)/video(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        num_samples = len(dataset)
        if "videos" in dataset[0]:
            media_type = "videos"
        else:
            media_type = "images"
        num_visuals = len(dataset[0][media_type])
        num_texts = len(dataset[0]['texts'])
        scores = torch.zeros(num_samples, num_visuals, num_texts).to(self.device)
        
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        counter = 0
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            cur_batch_size = len(batch[media_type][0])
            assert len(batch[media_type]) == num_visuals, \
                f"Number of visual (image/video) options in batch {batch_idx} is {len(batch[media_type])}. Expected {num_visuals} visuals."
            assert len(batch['texts']) == num_texts, \
                f"Number of text options in batch {batch_idx} is {len(batch['texts'])}. Expected {num_texts} texts."
            
            
            for vis_idx in range(num_visuals):
                visuals = batch[media_type][vis_idx]
                for text_idx in range(num_texts):
                    texts = batch['texts'][text_idx]
                    
                    if media_type == 'videos':
                        scores[counter:counter+cur_batch_size, vis_idx, text_idx] = \
                        torch.squeeze(torch.cat([self.forward(videos=visual, texts=text, num_frames=num_frames, **kwargs) for (visual, text) in zip(visuals, texts)], dim=0))
                    else:
                        scores[counter:counter+cur_batch_size, vis_idx, text_idx] = \
                        torch.squeeze(torch.cat([self.forward(images=visual, texts=text, **kwargs) for (visual, text) in zip(visuals, texts)], dim=0))
            
            counter += cur_batch_size
        return scores
    