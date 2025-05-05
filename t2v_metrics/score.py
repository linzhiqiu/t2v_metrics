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
                images: Optional[Union[str, List[str]]]=None,
                texts: Optional[Union[str, List[str]]]=None,
                num_frames: Optional[int]=8,
                **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s)/video(s) and the text(s)
        If there are m images/videos and n texts, return a m x n tensor
        
        Args:
            images: Path to image/video file, or list of paths to image frames
            texts: Text or list of texts to score against
            num_frames: Number of frames to extract from video
        """

        # Convert single inputs to lists
        if isinstance(images, str):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]

        delete_images = False
        processed_images = images

        # Handle video inputs for image-only models
        # Note: video processing for video-native models is handled at the model level
        valid_video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        if any(isinstance(img, str) and img[-4:].lower() in valid_video_extensions for img in images):
            if self.model.video_mode == "concat":
                delete_images = True
                processed_images = []
                
                for video in images:
                    # Handle video file or list of frames
                    frame_images = []
                    if isinstance(video, str):
                        output_dir = f"temp_{os.path.basename(video)}"
                        extract_frames(video, num_frames, output_dir)
                        frame_images = [cv2.imread(os.path.join(output_dir, f)) 
                                     for f in os.listdir(output_dir) if f.endswith('.jpg')]
                        
                        # Clean up temp directory
                        for f in os.listdir(output_dir):
                            os.remove(os.path.join(output_dir, f))
                        os.rmdir(output_dir)
                    elif isinstance(video, list):
                        frame_images = [cv2.imread(frame) for frame in video]

                    # Concatenate and save frames
                    concat_image = concatenate_images_horizontal(frame_images, dist_images=10)
                    output_path = f"concat_{os.path.basename(str(video))}.jpg"
                    cv2.imwrite(output_path, concat_image)
                    processed_images.append(output_path)
            elif self.model.video_mode != "direct":
                print(f"Invalid `video_mode` for the given model. Please check model's class attributes")
                return

        # Process scores
        scores = torch.zeros(len(processed_images), len(texts)).to(self.device)
        for i, image in enumerate(processed_images):
            scores[i] = self.model.forward([image] * len(texts), texts, **kwargs)
        
        # Cleanup temporary files
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
    