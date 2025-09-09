from .blip2_itm_model import BLIP2_ITM_MODELS, BLIP2ITMScoreModel
from .umt_itm_model import UMT_ITM_MODELS, UMTITMScoreModel
from .internvideo2_itm_model import INTERNVIDEO2_ITM_MODELS, InternVideo2ITMScoreModel
from ...constants import HF_CACHE_DIR

ALL_ITM_MODELS = [
    BLIP2_ITM_MODELS,
    UMT_ITM_MODELS,
    INTERNVIDEO2_ITM_MODELS,
]

def list_all_itmscore_models():
    return [model for models in ALL_ITM_MODELS for model in models]

def get_itmscore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR):
    assert model_name in list_all_itmscore_models()
    if model_name in BLIP2_ITM_MODELS:
        return BLIP2ITMScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in UMT_ITM_MODELS:
        return UMTITMScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in INTERNVIDEO2_ITM_MODELS:
        return InternVideo2ITMScoreModel(model_name, device=device, cache_dir=cache_dir)
    else:
        raise NotImplementedError()