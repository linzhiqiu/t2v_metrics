# import t2v_metrics

# llava_ov_score = t2v_metrics.VQAScore(model='llava-onevision-qwen2-7b-ov')
# image = "images/0.png" # an image path in string format
# text = "someone talks on the phone angrily while another person sits happily"
# score = llava_ov_score(images=[image], texts=[text])
# print(score)

import t2v_metrics
import torch
from transformers import AutoModelForCausalLM
# from transformers import mPLUGOwl3Config, mPLUGOwl3Model
from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu    # pip install decord


model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation='flash_attention_2', trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)
# add icecream and flash-attn
# model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
# config = mPLUGOwl3Config.from_pretrained(model_path)
# print(config)
# # model = mPLUGOwl3Model(config).cuda().half()
# model = mPLUGOwl3Model.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
# model.eval().cuda()
