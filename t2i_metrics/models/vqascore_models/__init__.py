from .clip_t5_model import CLIP_T5_MODELS, CLIPT5Model
from .llava_model import LLAVA_MODELS, LLaVAModel
from ...constants import HF_CACHE_DIR

ALL_VQA_MODELS = [
    CLIP_T5_MODELS,
    LLAVA_MODELS,
]

def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]

def get_vqascore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR):
    assert model_name in list_all_vqascore_models()
    if model_name in CLIP_T5_MODELS:
        return CLIPT5Model(model_name, device=device, cache_dir=cache_dir)
    elif model_name in LLAVA_MODELS:
        return LLaVAModel(model_name, device=device, cache_dir=cache_dir)
    else:
        raise NotImplementedError()