
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constants import HF_CACHE_DIR
from .vqascore import VQAScore, list_all_vqascore_models
from .clipscore import CLIPScore, list_all_clipscore_models
from .itmscore import ITMScore, list_all_itmscore_models

def list_all_models():
    return list_all_vqascore_models() + list_all_clipscore_models() + list_all_itmscore_models()

def get_score_model(model='clip-flant5-xxl', device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    if model in list_all_vqascore_models():
        return VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)
    elif model in list_all_clipscore_models():
        return CLIPScore(model, device=device, cache_dir=cache_dir, **kwargs)
    elif model in list_all_itmscore_models():
        return ITMScore(model, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()