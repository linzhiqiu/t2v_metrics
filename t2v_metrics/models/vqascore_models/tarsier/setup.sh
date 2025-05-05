#!/bin/bash
export AZURE_ENDPOINT=...
export OPENAI_API_KEY=...

pip3 install -r requirements.txt
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

# pip3 install httpx==0.23.0 # uncomment this line if found error in import openai
