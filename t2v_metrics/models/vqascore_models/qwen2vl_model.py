import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
from .vqa_model import VQAScoreModel

QWEN2_VL_MODELS = {
    # Qwen2_VL
    'qwen2-vl-2b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-2B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2-VL-2B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen2-vl-7b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen2-vl-72b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-72B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2-VL-72B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    # Qwen2.5_VL:
    'qwen2.5-vl-3b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-3B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'qwen2.5-vl-7b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },
    'qwen2.5-vl-32b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-32B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-32B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },
    'qwen2.5-vl-72b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-72B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-72B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0
    },

    # Winoground Finetuning
    'qwen2-vl-1': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/wino_lora_3epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-2': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/wino_lora_5epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-3': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/wino_dpo_3epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-4': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/wino_dpo_5epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    # GenAI-Bench Finetuning:

    'qwen2-vl-5': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/genai_lora_3epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-6': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/genai_lora_5epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-7': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/genai_dpo_3epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-8': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/genai_dpo_5epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },


    # NaturalBench Finetuning:

    'qwen2-vl-9': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/nb_lora_3epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-10': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/nb_lora_5epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-11': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/nb_dpo_3epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2-vl-12': {
        'tokenizer': {
            'path': 'Qwen/Qwen2-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2_vl-7b/lora/nb_dpo_5epochs',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    # Camera Motion Weights:
    'qwen2.5-vl-cam2500': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '../LLaMA-Factory/saves/qwen2.5_vl-7b/lora/cam_motion_sft_2500',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2.5-vl-cam10000': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-cam10000',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2.5-vl-cam15000': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-cam15000',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2.5-vl-balanced': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-balanced',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },

    'qwen2.5-vl-balanced2': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-balanced2',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
    },


   # Post-ICCV Results - set fps based on model name:
    'qwen2.5-vl-bal-cap-fps2': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-bal-cap-fps2',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 2.0,  # Specific fps from model name
    },

    'qwen2.5-vl-bal-cap-fps4': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-bal-cap-fps4',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 4.0,  # Specific fps from model name
    },

    'qwen2.5-vl-bal-cap-fps8': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-bal-cap-fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-bal-imb-cap-fps2': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-bal-imb-cap-fps2',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 2.0,  # Specific fps from model name
    },

    'qwen2.5-vl-bal-imb-cap-fps4': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-bal-imb-cap-fps4',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 4.0,  # Specific fps from model name
    },

    'qwen2.5-vl-bal-imb-cap-fps8': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-bal-imb-cap-fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-imb-cap-fps2': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-imb-cap-fps2',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 2.0,  # Specific fps from model name
    },

    'qwen2.5-vl-imb-fps2-full': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/full/sft/imb_fps2',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 2.0,  # Specific fps from model name
    },

    'qwen2.5-vl-imb-forward': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/lora/forward_fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-imb-backward': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/lora/backward_fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-imb-cap-fps8': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/lora/imb_cap_fps2',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-balraw': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/lora/balraw_cap_fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-imbraw': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/lora/imbraw_cap_fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-cam-centric-only': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/lora/cam_centric_only',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-rank64-5e-5': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/lora/bal_imb_cap_rank64_lr5e-5_fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-rank64-lr2e-4-freezevisTrue-2000it': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/saves/qwen2.5_vl-7b/lora/bal_imb_cap_rank64_lr2e-4_epoch10.0_freezevisTrue_fps8/checkpoint-2000',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-rank64-lr2e-4-freezevisTrue-800it': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/lora/bal_imb_cap_rank64_lr2e-4_epoch10.0_freezevisTrue_fps8/checkpoint-800',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },
    'qwen2.5-vl-full-lr2e-4-freezevisTrue-2000it': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/qwen2.5_vl-7b/full/sft/bal_imb_cap_full_lr2e-4_epoch10.0_freezevisTrue_fps8/checkpoint-2000',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-full-lr2e-4-freezevisTrue': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/saves/qwen2.5_vl-7b/full/sft/bal_imb_cap_full_lr2e-4_epoch10.0_freezevisTrue_fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },
    
    'qwen2.5-vl-cambench-sft-dpo-5000': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-7b-cambench5000-dpo',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-cambench-dpo-5000': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/models/qwen2.5-vl-7b-cambench5000-dpo',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },


    'qwen2.5-vl-32b-sft': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-32B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/saves/qwen2.5_vl-32b/lora/bal_imb_cap_rank64_lr1e-4_epoch2.0_freezevisTrue_fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-72b-sft': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-72B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/saves/qwen2.5_vl-72b/lora/bal_imb_cap_rank64_lr1e-4_epoch3.0_freezevisTrue_fps8/checkpoint-1800',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    # Caption RL Post Training:
    'qwen2.5-vl-7b-cam-motion-fft-ldpo': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/models/qwen2.5-vl-7b-cam-motion-fft-ldpo',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-7b-cam-motion-ldpo': {
            'tokenizer': {
                'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
            },
            'model': {
                'path': '/data3/cmitra/models/qwen2.5-vl-7b-cam-motion-ldpo',
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'flash_attention_2',
            },
            'fps': 8.0,  # Specific fps from model name
        },

    'qwen2.5-vl-1-3-6-lora': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/saves/qwen2.5-vl-7b-1-3-6',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-1-2-3-lora-epoch1': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/saves/qwen2.5_vl-7b/lora/bal_imb_cap_fps8',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },
    "train_all_except_proj": {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/saves_2/qwen2.5-vl-7b/full/train_all_except_proj/checkpoint-12000',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    }, 

    "train_proj_only": {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/saves_2/qwen2.5-vl-7b/train_proj_only/checkpoint-12800',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    }, 
    'qwen2.5-vl-7b-cambench': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': 'chancharikm/qwen2.5-vl-7b-cam-motion-preview',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },
    'qwen2.5-vl-32b-cambench': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': 'chancharikm/qwen2.5-vl-32b-cam-motion-preview',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'qwen2.5-vl-72b-cambench': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': 'chancharikm/qwen2.5-vl-72b-cam-motion-preview',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },

    'bal_imb_cap_fps_8': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': 'chancharikm/qwen2.5-vl-72b-cam-motion-preview',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,  # Specific fps from model name
    },
    #
    # Captioning FT Experiments
    #
    'qwen2.5-vl-hlc-ft-1': {
    'tokenizer': {
        'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
    },
    'model': {
        'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_1',
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-1-2-3-5-6-camera-only': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_1_2_3_5_6_camera_only',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-1-3': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_1_3',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-1-3-camera-only': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_1_3_camera_only',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-1-5': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_1_5',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-1-camera-only': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_1_camera_only',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-3': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_3',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-3-camera-only': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_3_camera_only',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-5': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_5',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2',
        },
        'fps': 8.0,
    },

    'qwen2.5-vl-hlc-ft-1-2-3-5-6': {
    'tokenizer': {
        'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
    },
    'model': {
        'path': '/data3/cmitra/LLaMA-Factory/saves/hlc_experiments/hlc_ft_1_2_3_5_6',
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_2',
    },
    'fps': 8.0,
},
}

class Qwen2VLModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='qwen2-vl-7b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in QWEN2_VL_MODELS, f"Model {model_name} not found in QWEN2_VL_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN2_VL_MODELS[model_name]
        self.load_model()

    def load_model(self):
        model_path = self.model_info['model']['path']
        if '2.5' in model_path:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info['model']['torch_dtype'],
                attn_implementation=self.model_info['model']['attn_implementation'],
                device_map="auto"
            )    
        
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info['model']['torch_dtype'],
                attn_implementation=self.model_info['model']['attn_implementation'],
                device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(self.model_info['tokenizer']['path'])
        self.model.eval()

        self.device = next(self.model.parameters()).device # If there are multiple GPUs put the model on the first parameters GPU

    def load_images(self, paths: List[str], num_frames: int = 16) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        fps = self.model_info.get('fps', 2.0)
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file path
                # video_frames = self.load_video(path, num_frames)
                processed_data.append({"type": "video", "video": path, "max_pixels": 360*420, "fps":fps})
            elif path.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    processed_data.append({"type": "image", "image": image})
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [Image.fromarray(frame.astype('uint8'), 'RGB') for frame in np_array]
                    processed_data.append({"type": "video", "video": frames})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                processed_data.append({"type": "image", "image": image})
        return processed_data

    def forward(self,
                paths: List[str],
                texts: List[str],
                num_frames: int=16,
                question_template: str = "Does this image show \"{}\"?", #"Does this image show \"{}\"? Answer the question with Yes or No",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(paths, num_frames)
        
        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
            messages = [{"role": "user", "content": [data, {"type": "text", "text": question}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False, # Odd that greedy decoding seems necessary for some reason to get the logprobs
                    output_scores=True,
                    return_dict_in_generate=True
                )


            scores = outputs.scores[0]

            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = self.processor.tokenizer.encode(answer)[0]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)
        return torch.tensor(lm_probs)
    
    def generate(self,
                images: List[str],
                texts: List[str],
                num_frames: int=16,
                max_new_tokens: int = 256) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        processed_data = self.load_images(images, num_frames)
        
        generated_texts = []
        for data, text in zip(processed_data, texts):
            messages = [{"role": "user", "content": [data, {"type": "text", "text": text}]}]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                generated_texts.append(text)
                
        return generated_texts