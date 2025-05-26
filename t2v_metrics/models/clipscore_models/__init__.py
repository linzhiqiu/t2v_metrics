from .clip_model import CLIP_MODELS, CLIPScoreModel
from .blip2_itc_model import BLIP2_ITC_MODELS, BLIP2ITCScoreModel
from .hpsv2_model import HPSV2_MODELS, HPSV2ScoreModel
from .pickscore_model import PICKSCORE_MODELS, PickScoreModel
from .umt_clip_model import UMT_CLIP_MODELS, UMTCLIPScoreModel
from .internvideo2_clip_model import INTERNVIDEO2_CLIP_MODELS, InternVideo2CLIPScoreModel
from .languagebind_video_clip_model import LANGUAGEBIND_VIDEO_CLIP_MODELS, LanguageBindVideoCLIPScoreModel
from ...constants import HF_CACHE_DIR

ALL_CLIP_MODELS = [
    CLIP_MODELS,
    BLIP2_ITC_MODELS,
    HPSV2_MODELS,
    PICKSCORE_MODELS,
    UMT_CLIP_MODELS,
    INTERNVIDEO2_CLIP_MODELS,
    LANGUAGEBIND_VIDEO_CLIP_MODELS,
]

def list_all_clipscore_models():
    return [model for models in ALL_CLIP_MODELS for model in models]

def get_clipscore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_clipscore_models()
    if model_name in CLIP_MODELS:
        return CLIPScoreModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in BLIP2_ITC_MODELS:
        return BLIP2ITCScoreModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in HPSV2_MODELS:
        return HPSV2ScoreModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in PICKSCORE_MODELS:
        return PickScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in UMT_CLIP_MODELS:
        return UMTCLIPScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in INTERNVIDEO2_CLIP_MODELS:
        return InternVideo2CLIPScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in LANGUAGEBIND_VIDEO_CLIP_MODELS:
        return LanguageBindVideoCLIPScoreModel(model_name, device=device, cache_dir=cache_dir)
    else:
        raise NotImplementedError()
