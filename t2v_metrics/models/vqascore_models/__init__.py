from .gpt4v_model import GPT4V_MODELS, GPT4VModel
from .paligemma_model import PALIGEMMA_MODELS, PaliGemmaModel
# from .internvl_model import INTERNVL_MODELS, InternVLModel
from .gemini_model import GEMINI_MODELS, GeminiModel
from .qwen2vl_model import QWEN2_VL_MODELS, Qwen2VLModel
from .qwen3vl_model import QWEN3_VL_MODELS, Qwen3VLModel
from .qwen3omni_model import QWEN3_OMNI_MODELS, Qwen3OmniModel
from .gemma3_model import GEMMA3_MODELS, Gemma3Model
from .gemma4_model import GEMMA4_MODELS, Gemma4Model
# from .molmo2_model import MOLMO2_MODELS, Molmo2Model

from ...constants import HF_CACHE_DIR

ALL_VQA_MODELS = [
    GPT4V_MODELS,
    PALIGEMMA_MODELS,
    # INTERNVL_MODELS,
    GEMINI_MODELS,
    QWEN2_VL_MODELS,
    QWEN3_VL_MODELS,
    QWEN3_OMNI_MODELS,
    GEMMA3_MODELS,
    GEMMA4_MODELS,
    # MOLMO2_MODELS
]


def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]

def get_vqascore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models()
    if model_name in GPT4V_MODELS:
       return GPT4VModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in PALIGEMMA_MODELS:
        return PaliGemmaModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    # elif model_name in INTERNVL_MODELS:
    #     return InternVLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in GEMINI_MODELS:
        return GeminiModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in QWEN2_VL_MODELS:
        return Qwen2VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in QWEN3_VL_MODELS:
        return Qwen3VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in QWEN3_OMNI_MODELS:
        return Qwen3OmniModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in GEMMA3_MODELS:
        return Gemma3Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in GEMMA4_MODELS:
        return Gemma4Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    # elif model_name in MOLMO2_MODELS:
    #     return Molmo2Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()