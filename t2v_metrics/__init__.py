from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

import subprocess
import shutil

# Check for ffmpeg on import - REQUIRED system dependency
try:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    raise RuntimeError(
        "ffmpeg is a required system requirement but not found. Install with:\n"
        "conda install ffmpeg=6.1.2 -c conda-forge\n"
        "or visit: https://ffmpeg.org/download.html"
    )

from .constants import HF_CACHE_DIR
from .vqascore import VQAScore, list_all_vqascore_models


def list_all_models():
    return list_all_vqascore_models() 

def get_score_model(model='clip-flant5-xxl', device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    if model in list_all_vqascore_models():
        return VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()