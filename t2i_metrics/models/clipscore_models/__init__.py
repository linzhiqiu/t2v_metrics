from .clip_model import CLIP_MODELS, CLIPScoreModel
from ...constants import HF_CACHE_DIR

ALL_CLIP_MODELS = [
    CLIP_MODELS,
]

def list_all_clipscore_models():
    return [model for models in ALL_CLIP_MODELS for model in models]

def get_clipscore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR):
    assert model_name in list_all_clipscore_models()
    if model_name in CLIP_MODELS:
        return CLIPScoreModel(model_name, device=device, cache_dir=cache_dir)
    else:
        raise NotImplementedError()