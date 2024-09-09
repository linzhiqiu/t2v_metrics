from typing import List
import torch
import sys
import os
from transformers import AutoTokenizer, GenerationConfig
from ...constants import HF_CACHE_DIR

# Add the ViLA package path as a variable for flexibility
VILA_PACKAGE_PATH = '/home/zhaobin/VILA'
sys.path.append(VILA_PACKAGE_PATH)

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, process_images, tokenizer_image_token, get_model_name_from_path)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from .vqa_model import VQAScoreModel

default_question_template = 'Question: Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "yes"

VILA_MODELS = {
    'ViLA': {'ckpt': 'Efficient-Large-Model/Llama-3-VILA1.5-8b'}
}

class ViLAModel(VQAScoreModel):
    """A wrapper for the ViLA models"""
    def __init__(self,
                 model_name='ViLA',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        assert model_name in VILA_MODELS, f"Model name {model_name} not found in VILA_MODELS"
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image processor"""
        ckpt = VILA_MODELS[self.model_name]['ckpt']
        disable_torch_init()
        model_name = get_model_name_from_path(ckpt)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(ckpt, model_name, None)
        self.model = self.model.eval().to(self.device)

    def insert_image(self, text, image_list):
        """Insert images into the text using ViLA-specific formatting"""
        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images_tensor = process_images(load_images(image_list), self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        return (input_ids, images_tensor, stopping_criteria, stop_str)

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, images: List[str], texts: List[str], question_template: str = default_question_template, answer_template: str = default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)"""
        assert len(images) == len(texts), "Number of images and texts must match"
        questions = [question_template.format(text) for text in texts]
        model_inputs = self.insert_image(questions[0], images)
        outputs = self.model(model_inputs[0], images=[model_inputs[1]])
        logits = outputs.logits
        tokenized_answers = self.tokenizer([answer_template] * len(texts), return_tensors='pt', padding='longest')
        labels = tokenized_answers['input_ids'].to('cuda')
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        lm_prob = torch.zeros(outputs.logits.shape[0])
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp()
        return lm_prob
