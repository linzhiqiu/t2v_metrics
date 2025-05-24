from .clip_t5_model import CLIP_T5_MODELS, CLIPT5Model
from .llava_model import LLAVA_MODELS, LLaVAModel
from .llava16_model import LLAVA16_MODELS, LLaVA16Model
from .instructblip_model import InstructBLIP_MODELS, InstructBLIPModel
from .gpt4v_model import GPT4V_MODELS, GPT4VModel
from .llavaov_model import LLAVA_OV_MODELS, LLaVAOneVisionModel
from .mplug_model import MPLUG_OWL3_MODELS, mPLUGOwl3Model
from .paligemma_model import PALIGEMMA_MODELS, PaliGemmaModel
from .internvl_model import INTERNVL2_MODELS, InternVL2Model
from .internvideo_model import INTERNVIDEO2_MODELS, InternVideo2Model
from .internlm_model import INTERNLMXCOMPOSER25_MODELS, InternLMXComposer25Model
from .llama32_model import LLAMA_32_VISION_MODELS, LLaMA32VisionModel
from .molmo_model import MOLMO_MODELS, MOLMOVisionModel
from .gemini_model import GEMINI_MODELS, GeminiModel
from .qwen2vl_model import QWEN2_VL_MODELS, Qwen2VLModel
from .llavavideo_model import LLAVA_VIDEO_MODELS, LLaVAVideoModel
from .tarsier_model import TARSIER_MODELS, TarsierModel
from .perceptionlm_model import PERCEPTION_LM_MODELS, PerceptionLMModel

from ...constants import HF_CACHE_DIR

ALL_VQA_MODELS = [
    CLIP_T5_MODELS,
    LLAVA_MODELS,
    LLAVA16_MODELS,
    InstructBLIP_MODELS,
    GPT4V_MODELS,
    LLAVA_OV_MODELS,
    MPLUG_OWL3_MODELS,
    PALIGEMMA_MODELS,
    INTERNVL2_MODELS,
    INTERNVIDEO2_MODELS,
    INTERNLMXCOMPOSER25_MODELS,
    LLAMA_32_VISION_MODELS,
    MOLMO_MODELS,
    GEMINI_MODELS,
    QWEN2_VL_MODELS,
    LLAVA_VIDEO_MODELS,
    TARSIER_MODELS,
    PERCEPTION_LM_MODELS
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
        return InstructBLIPModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in GPT4V_MODELS:
        return GPT4VModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in LLAVA_OV_MODELS:
        return LLaVAOneVisionModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in MPLUG_OWL3_MODELS:
        return mPLUGOwl3Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in PALIGEMMA_MODELS:
        return PaliGemmaModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in INTERNVL2_MODELS:
        return InternVL2Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in INTERNVIDEO2_MODELS:
        return InternVideo2Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in INTERNLMXCOMPOSER25_MODELS:
        return InternLMXComposer25Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in LLAMA_32_VISION_MODELS:
        return LLaMA32VisionModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in MOLMO_MODELS:
        return MOLMOVisionModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in GEMINI_MODELS:
        return GeminiModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in QWEN2_VL_MODELS:
        return Qwen2VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in LLAVA_VIDEO_MODELS:
        return LLaVAVideoModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in TARSIER_MODELS:
        return TarsierModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in PERCEPTION_LM_MODELS:
        return PerceptionLMModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()