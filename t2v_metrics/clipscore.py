from typing import List

from .score import Score

from .constants import HF_CACHE_DIR

from .models.clipscore_models import list_all_clipscore_models, get_clipscore_model

class CLIPScore(Score):
    def prepare_scoremodel(self,
                           model='openai:ViT-L/14',
                           device='cuda',
                           cache_dir=HF_CACHE_DIR,
                           **kwargs):
        print(f'{kwargs["model_path"]}')
        return get_clipscore_model(
            model,
            device=device,
            cache_dir=cache_dir,
            **kwargs
        )
            
    def list_all_models(self) -> List[str]:
        return list_all_clipscore_models()