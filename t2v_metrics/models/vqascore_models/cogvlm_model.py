from typing import List
import torch
import copy
from PIL import Image

from .vqa_model import VQAScoreModel
from ...constants import HF_CACHE_DIR
from transformers import AutoModelForCausalLM, LlamaTokenizer

CONTEXT_LEN = 2048
IGNORE_INDEX = -100

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

def format_question(question, conversation_style='chat'):
    if conversation_style == 'plain':
        question = question
    elif conversation_style == 'chat':
        question = question
    else:
        raise NotImplementedError()
    return question

def format_answer(answer, conversation_style='chat'):
    if conversation_style == 'plain':
        answer = answer + "\n"
    elif conversation_style == 'chat':
        answer = answer + "</s>"
    else:
        raise NotImplementedError()
    return answer

COGVLM_MODELS = {
    'cogvlm-17b': {
        'tokenizer' : {
            'path': 'lmsys/vicuna-7b-v1.5',
        },
        'model': {
            'path': 'THUDM/cogvlm-chat-hf',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
}


class CogVLMModel(VQAScoreModel):
    """A wrapper for the CogVLM model"""
    def __init__(self,
                 model_name='cogvlm-17b',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        assert model_name in COGVLM_MODELS
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """

        self.conversational_style = COGVLM_MODELS[self.model_name]['model']['conversation']
        
        self.context_len = CONTEXT_LEN

        self.tokenizer = LlamaTokenizer.from_pretrained(COGVLM_MODELS[self.model_name]['tokenizer']['path'], cache_dir=HF_CACHE_DIR, add_bos_token=False)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            COGVLM_MODELS[self.model_name]['model']['path'],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            ).to(self.device).eval()

    def load_images(self, image_paths: List[str]) -> torch.Tensor:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        return images

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

        # Format questions and answers
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]

        # Load images
        images = self.load_images(images)

        # Build input ids for the question
        q = self.model.build_conversation_input_ids(self.tokenizer, query=questions[0], history=[], images=[images[0]], template_version=self.conversational_style)

        question_len = len(q['input_ids'])

        # Append token ids of the answer at the end
        tokens_to_append = self.tokenizer.encode(answers[0])
        q['input_ids']=torch.hstack([q['input_ids'],torch.tensor(tokens_to_append)])
        q['token_type_ids'] = torch.hstack([q['token_type_ids'],torch.tensor([0]*len(tokens_to_append))])
        q['attention_mask'] = torch.hstack([q['attention_mask'],torch.tensor([1]*len(tokens_to_append))])

        # Create labels ids and mask question tokens
        labels = copy.deepcopy(q['input_ids'])
        labels[:question_len] = IGNORE_INDEX

        # Prepare inputs and labels for model
        inputs = {
            'input_ids': q['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': q['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': q['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[q['images'][0].to('cuda').to(torch.bfloat16)]],
            'labels': labels.unsqueeze(0).to('cuda'),
        }

        outputs = self.model(**inputs, return_dict=True)

        # Compute VQAScore from the loss
        loss = outputs.loss
        lm_prob = (-loss).exp()
        return lm_prob



    




