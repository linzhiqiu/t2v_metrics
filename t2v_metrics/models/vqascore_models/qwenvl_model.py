from typing import List
import torch
import copy
import os
from torchvision import transforms

from .vqa_model import VQAScoreModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from ...constants import HF_CACHE_DIR

default_question_template = 'Question: Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "yes" # instruct-blip uses "yes" instead of "Yes"

QwenVL_MODELS = {
    'Qwen-VL': {'ckpt': 'Qwen/QwenVL'},
    'Qwen-VL-Chat': {'ckpt': 'Qwen/Qwen-VL-Chat'},
}

class QwenVLModel(VQAScoreModel):
    """A wrapper for the InstructBLIP (FlanT5-based) models"""
    def __init__(self,
                 model_name='Qwen-VL',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        assert model_name in QwenVL_MODELS, f"Model name {model_name} not found in QwenVL_MODELS"
        # os.environ['TORCH_HOME'] = cache_dir
        # import timm.models.hub as timm_hub
        # if cache_dir != timm_hub.get_cache_dir():
        #     print(f"Warning: cache_dir {cache_dir} won't be used because "
        #            "InstructBLIP is cached using timm_hub.get_cache_dir(). "
        #            "You may want to update os.environ['TORCH_HOME'] before importing to specify the cache_dir.")
        #     print(f"Model saved to {timm_hub.get_cache_dir()}")
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """

        ckpt = QwenVL_MODELS[self.model_name]['ckpt']
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
        # use cuda device
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto", trust_remote_code=True).eval()

        # Specify hyperparameters for generation
        self.model.generation_config = GenerationConfig.from_pretrained(ckpt, trust_remote_code=True)

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        image = [self.image_preprocess(image) for image in image]
        assert all(x.shape == image[0].shape for x in image)
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str=default_question_template,
                answer_template: str=default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Is the image showing 'a photo of a dog'? Please answer yes or no."
        # A: "Yes"
        questions = ["<img>" + img + "</img> " + question_template.format(text) for text, img in zip(texts, images)]
        tokenized_questions = self.tokenizer(questions, return_tensors='pt', padding='longest')
        answers = [answer_template.format(text) + "<|endoftext|>" for text in texts]
        tokenized_answers = self.tokenizer(answers, return_tensors='pt', padding='longest')

        # print(f'Questions {questions} \n\n Answers {answers}')
        # exit()
        
        print(answers)
        print(tokenized_answers)
        outputs = self.model(input_ids=tokenized_questions['input_ids'].to('cuda'), attention_mask=tokenized_questions['attention_mask'], labels=tokenized_answers['input_ids'])
        

        print(f'Model Loss {outputs.loss} Loss Shape {outputs.loss.shape}')
        logits = outputs.logits
        labels = tokenized_answers

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        lm_prob = torch.zeros(outputs.logits.shape[0])
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp()
        return lm_prob