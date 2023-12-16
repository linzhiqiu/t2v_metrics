from abc import abstractmethod
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from .constants import HF_CACHE_DIR

class Score(nn.Module):

    def __init__(self,
                 model: str,
                 device: str='cuda',
                 cache_dir: str=HF_CACHE_DIR):
        """Initialize the ScoreModel
        """
        super().__init__()
        assert model in self.list_all_models()
        self.device = device
        self.model = self.prepare_scoremodel(model, device, cache_dir)
    
    @abstractmethod
    def prepare_scoremodel(self,
                           model: str,
                           device: str,
                           cache_dir: str):
        """Prepare the ScoreModel
        """
        pass
    
    @abstractmethod
    def list_all_models(self) -> List[str]:
        """List all available models
        """
        pass

    def forward(self,
                images: List[str],
                texts: List[str],
                **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        scores = torch.zeros(len(images), len(texts)).to(self.device)
        for i, image in enumerate(images):
            scores[i] = self.model.forward([image] * len(texts), texts, **kwargs)
        return scores
    