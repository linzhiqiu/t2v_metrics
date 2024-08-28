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
    """A wrapper for the QwenVL models"""
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
<<<<<<< HEAD
        self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
  
=======
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
>>>>>>> b79f41867c9f4525a66f7bb88b4895d9767c094c
        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
        # use cuda device
<<<<<<< HEAD
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, device_map="cuda", trust_remote_code=True).eval()
        self.model.resize_token_embeddings(len(self.tokenizer))
=======
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto", trust_remote_code=True).eval()

>>>>>>> b79f41867c9f4525a66f7bb88b4895d9767c094c
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
<<<<<<< HEAD
        tokenized_questions = self.tokenizer(questions, return_tensors='pt', padding='longest').to(self.device)
        # unpadded_questions = self.tokenizer(questions[0], return_tensors='pt', padding='do_not_pad').to(self.device)
        print(f'Padded Question {self.tokenizer.batch_decode(tokenized_questions["input_ids"])[0]} Attention Mask {tokenized_questions["attention_mask"][0]}')
        answers = [answer_template.format(text) + '<|endoftext|>' for text in texts]
        # print(questions)
        tokenized_answers = self.tokenizer(answers, return_tensors='pt', padding='longest').to(self.device)
        # print(f'Questions {questions} \n\n Answers {answers}')
        # exit()
        
        # print(tokenized_questions)
        # print(len(self.tokenizer.batch_decode(tokenized_questions['input_ids'])[0]))
        # input_lengths = [len(i) for i in tokenized_questions['input_ids']]
        # print(input_lengths)
        outputs = self.model(input_ids=tokenized_questions['input_ids'], attention_mask=tokenized_questions['attention_mask'])
        # outputs = self.model.generate(input_ids=tokenized_questions['input_ids'], attention_mask=tokenized_questions['attention_mask'],
        # pad_token_id=self.tokenizer.eod_id, eos_token_id=self.tokenizer.eod_id)


        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        print(f'Logits {self.tokenizer.batch_decode(predicted_token_ids)[0]}')
        # print(f'Outputs {self.tokenizer.batch_decode(outputs)[0]}')
=======
        tokenized_questions = self.tokenizer(questions, return_tensors='pt', padding='longest')
        answers = [answer_template.format(text) + "<|endoftext|>" for text in texts]
        tokenized_answers = self.tokenizer(answers, return_tensors='pt', padding='longest')

        # print(f'Questions {questions} \n\n Answers {answers}')
        # exit()
        
        print(answers)
        print(tokenized_answers)
        outputs = self.model(input_ids=tokenized_questions['input_ids'].to('cuda'), attention_mask=tokenized_questions['attention_mask'], labels=tokenized_answers['input_ids'])
        

        print(f'Model Loss {outputs.loss} Loss Shape {outputs.loss.shape}')
>>>>>>> b79f41867c9f4525a66f7bb88b4895d9767c094c
        logits = outputs.logits
        
        last_non_padding_idx = tokenized_questions['attention_mask'].sum(dim=1) - 1
        print(last_non_padding_idx)
        labels = tokenized_answers

        # print(tokenized_questions)

        #print(self.tokenizer.batch_decode(outputs)[0])
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        lm_prob = torch.zeros(outputs.logits.shape[0])
        for k in range(lm_prob.shape[0]): # which is just batch size
            # input_len = input_lengths[k]
            # print(self.tokenizer.batch_decode(tokenized_questions['input_ids']))
            # print(lm_prob.shape)
            # print(logits.shape)
            # print(torch.flatten(labels['input_ids']).shape)
            
            # print(input_len)
            print(labels['input_ids'].shape)
            relevant_logits = logits[:, last_non_padding_idx[k]:last_non_padding_idx[k] + 2, :]
            print(relevant_logits.shape)
            lm_prob[k] = (-loss_fct(relevant_logits[k], labels['input_ids'][k])).exp()
        return lm_prob

