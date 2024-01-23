from typing import List
import torch
import copy

from .vqa_model import VQAScoreModel
from ...constants import HF_CACHE_DIR, IGNORE_INDEX, CONTEXT_LEN
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

default_question_template = "Is the image showing '{}'? Please answer yes or no."
default_answer_template = "Yes"

KOSMOS2_MODELS = {
    'kosmos-2-patch14-224': {
        'checkpoint_path': 'microsoft/kosmos-2-patch14-224',
        'model_max_length': CONTEXT_LEN
    }
}

def format_question(question, kosmos2_vqa_template=False):
    if kosmos2_vqa_template:
        return '<grounding> Question: ' + question
    else:
        return question

def format_answer(answer, kosmos2_vqa_template=False):
    if kosmos2_vqa_template:
        return 'Answer: ' + answer
    else:
        return answer

class Kosmos2Model(VQAScoreModel):
    """A wrapper for the KOSMOS-2 model"""
    def __init__(self,
                 model_name='kosmos-2-patch14-224',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        assert model_name in KOSMOS2_MODELS
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)
    def load_model(self):
        """Load the model, tokenizer, and processor
        """
        self.model = Kosmos2ForConditionalGeneration.from_pretrained(
            KOSMOS2_MODELS[self.model_name]['checkpoint_path'],
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            KOSMOS2_MODELS[self.model_name]['checkpoint_path'],
            cache_dir=self.cache_dir
        )
        self.model.to(self.device)
        self.model.requires_grad_(False)
        self.tokenizer = self.processor.tokenizer
        self.context_len = KOSMOS2_MODELS[self.model_name]['model_max_length']

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s)
        """
        size = 224
        image = [self.image_loader(x) for x in image]
        image = [x.resize((size, size)) for x in image]
        # Processor do all the image preprocessing and expects PIL image
        return image

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)
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
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        # Format the questions and answers
        questions = [format_question(question) for question in questions]
        answers = [format_answer(answer) for answer in answers]

        prompts =  [question + answer for question, answer in zip(questions, answers)]

        images = self.load_images(images)

        # Encode the prompts
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.context_len
        ).to(self.device)

        input_ids = inputs["input_ids"].to(self.device)
        labels = copy.deepcopy(input_ids)
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        # Forward through the model
        outputs = self.model(**inputs, labels=labels)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        output_logits = outputs.logits[:, :-1, :].contiguous()
        output_labels = labels[:, 1:].contiguous()
        lm_prob = torch.zeros(output_logits.shape[0])
        for i in range(lm_prob.shape[0]):
            lm_prob[i] = (-loss_fct(output_logits[i], output_labels[i])).exp()

        return lm_prob