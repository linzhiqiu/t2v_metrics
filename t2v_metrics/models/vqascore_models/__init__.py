from .clip_t5_model import CLIP_T5_MODELS, CLIPT5Model
from .llava_model import LLAVA_MODELS, LLaVAModel
from .llava16_model import LLAVA16_MODELS, LLaVA16Model
from .instructblip_model import InstructBLIP_MODELS, InstructBLIPModel
from .qwenvl_model import QwenVL_MODELS, QwenVLModel
from .gpt4v_model import GPT4V_MODELS, GPT4VModel
from .llavaov_model import LLAVA_OV_MODELS, LLaVAOneVisionModel
from ...constants import HF_CACHE_DIR

ALL_VQA_MODELS = [
    CLIP_T5_MODELS,
    LLAVA_MODELS,
    LLAVA16_MODELS,
    InstructBLIP_MODELS,
    QwenVL_MODELS,
    GPT4V_MODELS,
    LLAVA_OV_MODELS
]

def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]

def get_vqascore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models()
    if model_name in CLIP_T5_MODELS:
        return CLIPT5Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in LLAVA_MODELS:
        return LLaVAModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in LLAVA16_MODELS:
        return LLaVA16Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in InstructBLIP_MODELS:
        return InstructBLIPModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in QwenVL_MODELS:
        return QwenVLModel(model_name, device=device, cache_dir=cache_dir)
        # return InstructBLIPModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in GPT4V_MODELS:
        return GPT4VModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in LLAVA_OV_MODELS:
        return LLaVAOneVisionModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()