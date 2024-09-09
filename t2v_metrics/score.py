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
                concatenate: Optional[str]=None,
                images: Optional[Union[str, List[str]]]=None,
                texts: Optional[Union[str, List[str]]]=None,
                **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s)/video(s) and the text(s)
        If there are m images/videos and n texts, return a m x n tensor
        """
        if videos is not None:
            if isinstance(videos, str):
                videos = [videos]
            
            
            if any(name in self.model_name.lower() for name in ['clip', 'blip', 'llava-v1.5', 'llava-v1.6', 'hpsv2', 'pickscore', 'imag-reward']):
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
            else:
                images = videos
        
        if isinstance(images, str):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]
        
        scores = torch.zeros(len(images), len(texts)).to(self.device)
        for i, image in enumerate(images):
            scores[i] = self.model.forward([image] * len(texts), texts, **kwargs)
        return scores
    # def forward(self,
    #             videos: Optional[Union[str, List[str]]],
    #             num_frames: Optional[Union[str, List[str]]],
    #             concatenate: Optional[str],
    #             images: Optional[Union[str, List[str]]],
    #             texts: Optional[Union[str, List[str]]],
    #             **kwargs) -> torch.Tensor:
    #     """Return the similarity score(s) between the image(s) and the text(s)
    #     If there are m images and n texts, return a m x n tensor
    #     """
    #     if type(images) == str:
    #         images = [images]
    #     if type(texts) == str:
    #         texts = [texts]
        
    #     scores = torch.zeros(len(images), len(texts)).to(self.device)
    #     for i, image in enumerate(images):
    #         scores[i] = self.model.forward([image] * len(texts), texts, **kwargs)
    #     return scores
    
    def batch_forward(self,
                      dataset: List[ImageTextDict],
                      batch_size: int=16,
                      **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        num_samples = len(dataset)
        num_images = len(dataset[0]['images'])
        num_texts = len(dataset[0]['texts'])
        scores = torch.zeros(num_samples, num_images, num_texts).to(self.device)
        
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        counter = 0
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            cur_batch_size = len(batch['images'][0])
            assert len(batch['images']) == num_images, \
                f"Number of image options in batch {batch_idx} is {len(batch['images'])}. Expected {num_images} images."
            assert len(batch['texts']) == num_texts, \
                f"Number of text options in batch {batch_idx} is {len(batch['texts'])}. Expected {num_texts} texts."
            
            for image_idx in range(num_images):
                images = batch['images'][image_idx]
                for text_idx in range(num_texts):
                    texts = batch['texts'][text_idx]
                    scores[counter:counter+cur_batch_size, image_idx, text_idx] = \
                        self.model.forward(images, texts, **kwargs)
            
            counter += cur_batch_size
        return scores
    