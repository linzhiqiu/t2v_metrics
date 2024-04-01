from typing import List

from .score import Score

from .constants import HF_CACHE_DIR

from .models.itmscore_models import list_all_itmscore_models, get_itmscore_model

class ITMScore(Score):
    def prepare_scoremodel(self,
                           model='blip2-itm',
                           device='cuda',
                           cache_dir=HF_CACHE_DIR):
        return get_itmscore_model(
            model,
            device=device,
            cache_dir=cache_dir
        )
            
    def list_all_models(self) -> List[str]:
        return list_all_itmscore_models()