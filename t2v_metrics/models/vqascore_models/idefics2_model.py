from typing import List
import torch
import sys
from transformers import AutoProcessor, AutoModelForVision2Seq
from ...constants import HF_CACHE_DIR
from .vqa_model import VQAScoreModel

default_question_template = 'Question: Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "yes"

IDEFICS2_MODELS = {
    'Idefics2': {'ckpt': 'HuggingFaceM4/idefics2-8b'}
}

class Idefics2Model(VQAScoreModel):
    """A wrapper for the Idefics2 models"""
    def __init__(self,
                 model_name='Idefics2',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        assert model_name in IDEFICS2_MODELS, f"Model name {model_name} not found in IDEFICS2_MODELS"
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        """Load the model, processor"""
        ckpt = IDEFICS2_MODELS[self.model_name]['ckpt']
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.processor.image_processor.do_image_splitting = False
        self.model = AutoModelForVision2Seq.from_pretrained(
            ckpt,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.model.eval()

    def insert_image(self, text, image_list):
        """Insert images into the text using Idefics2-specific formatting"""
        opened_images = [Image.open(image).convert("RGB") for image in image_list]
        inputs = self.processor(text=[text], images=[opened_images], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, images: List[str], texts: List[str], question_template: str = default_question_template, answer_template: str = default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)"""
        assert len(images) == len(texts), "Number of images and texts must match"
        questions = [question_template.format(text) for text in texts]
        model_inputs = self.insert_image(questions[0], images)
        outputs = self.model(**model_inputs)
        logits = outputs.logits
        tokenized_answers = self.processor.tokenizer([answer_template] * len(texts), return_tensors='pt', padding='longest')
        labels = tokenized_answers['input_ids'].to(self.device)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        lm_prob = torch.zeros(logits.shape[0])
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp()
        return lm_prob
