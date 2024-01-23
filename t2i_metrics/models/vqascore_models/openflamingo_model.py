from typing import List
import torch
import copy
import os
from torchvision import transforms

from .vqa_model import VQAScoreModel
from ...constants import HF_CACHE_DIR, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, END_OF_CHUNK, CONTEXT_LEN
from .open_flamingo import create_model_and_transforms

default_question_template = "Is the image showing '{}'? Please answer yes or no."
default_answer_template = "Yes"

def format_vqa_prompt(question, answer, openflamingo_vqa_template=False):
    """
    Format the question and answer into a prompt for OpenFlamingo.
    OpenFlamingo VQA prompt: '<image>Question: <question> Short answer: <answer> <END_OF_CHUNK>'
    """
    if openflamingo_vqa_template:
        question = DEFAULT_IMAGE_TOKEN + 'Question: ' + question
        answer = 'Short answer: ' + answer + END_OF_CHUNK
    else:
        question = DEFAULT_IMAGE_TOKEN + question
        answer = answer + END_OF_CHUNK

    return question + ' ' + answer

def prepare_text(
    batch: List[str],
    padding='longest',
    truncation=True,
    max_length=2048,
    tokenizer=None,
    device="cuda",
    ):
    """
    Tokenize the texts in a batch and 
    return the input_ids and attention_mask tensors.
    """
    encodings = tokenizer(
        batch,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
        max_length=max_length,
    )
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(
        device, non_blocking=True
    )
    return input_ids, attention_mask

OPEN_FLAMINGO_MODELS = {
    'open_flamingo_3b': {
        'tokenizer': {
            'path': 'anas-awadalla/mpt-1b-redpajama-200b',
            'model_max_length': 2048,
        },
        'checkpoint_path': 'openflamingo/OpenFlamingo-3B-vitl-mpt1b', 
        'clip_vision_encoder': {
            'path': 'ViT-L-14',
            'pretrained': 'openai'
        },
        'lang_encoder': 'anas-awadalla/mpt-1b-redpajama-200b',
        'cross_attn_every_n_layers': 1
    },
    'open_flamingo_3b_instruct': {
        'tokenizer': {
            'path': 'anas-awadalla/mpt-1b-redpajama-200b-dolly',
            'model_max_length': 2048,
        },
        'checkpoint_path': 'openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct', 
        'clip_vision_encoder': {
            'path': 'ViT-L-14',
            'pretrained': 'openai'
        },
        'lang_encoder': 'anas-awadalla/mpt-1b-redpajama-200b-dolly',
        'cross_attn_every_n_layers': 1
    },
    'open_flamingo_4b': {
        'tokenizer': {
            'path': 'togethercomputer/RedPajama-INCITE-Base-3B-v1',
            'model_max_length': 2048,
        },
        'checkpoint_path': 'openflamingo/OpenFlamingo-4B-vitl-rpj3b', 
        'clip_vision_encoder': {
            'path': 'ViT-L-14',
            'pretrained': 'openai'
        },
        'lang_encoder': 'togethercomputer/RedPajama-INCITE-Base-3B-v1',
        'cross_attn_every_n_layers': 2
    },
    'open_flamingo_4b_instruct': {
        'tokenizer': {
            'path': 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1',
            'model_max_length': 2048,
        },
        'checkpoint_path': 'openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct', 
        'clip_vision_encoder': {
            'path': 'ViT-L-14',
            'pretrained': 'openai'
        },
        'lang_encoder': 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1',
        'cross_attn_every_n_layers': 2
    },
    # OpenFlamingo 9B model requires transformers>=4.28.1(currently not supported)
    # 'open_flamingo_9b': {
    #     'tokenizer': {
    #         'path': 'anas-awadalla/mpt-7b',
    #         'model_max_length': 2048,
    #     },
    #     'checkpoint_path': 'openflamingo/OpenFlamingo-9B-vitl-mpt7b', 
    #     'clip_vision_encoder': {
    #         'path': 'ViT-L-14',
    #         'pretrained': 'openai'
    #     },
    #     'lang_encoder': 'anas-awadalla/mpt-7b',
    #     'cross_attn_every_n_layers': 4
    # }, 
}


class OpenFlamingoModel(VQAScoreModel):
    """A wrapper for the OpenFlamingo models"""
    def __init__(self,
                 model_name='open_flamingo_3b',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        assert model_name in OPEN_FLAMINGO_MODELS
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)
    
    def load_model(self):
        model_config = OPEN_FLAMINGO_MODELS[self.model_name]
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            model_config['clip_vision_encoder']['path'],
            model_config['clip_vision_encoder']['pretrained'],
            model_config['lang_encoder'],
            model_config['tokenizer']['path'],
            model_config['cross_attn_every_n_layers'],
        )
        model_base_name = model_config['checkpoint_path'].split('/')[-1]
        local_dir = os.path.join(self.cache_dir, model_base_name)

        # Check if "checkpoint.pt" exists in the local_dir before downloading weights
        if os.path.exists(os.path.join(local_dir, 'checkpoint.pt')):
            checkpoint = os.path.join(local_dir, 'checkpoint.pt')
        else:
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download(model_config['checkpoint_path'], 'checkpoint.pt', local_dir=local_dir)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.context_len = model_config['tokenizer']['model_max_length']

    def load_images(self, images: List[str]) -> torch.Tensor:
        """
        Load the images, preprocess them, and return a batch tensor on self.device.
        The returned image tensor has shape (B, T_img, F, C, H, W).
        T_img(no of images in one sample) and F(frames) are 1.
        """
        batch = [self.image_loader(x) for x in images]
        images_per_example = 1  
        frames_per_image = 1  

        first_image = self.image_processor(batch[0])
        batch_images = torch.zeros(
            (len(batch), images_per_example, frames_per_image, *first_image.shape),
            dtype=first_image.dtype,
            device=self.device  
        )
        for iexample, example in enumerate(batch):
            preprocessed = self.image_processor(example)
            batch_images[iexample, 0, 0] = preprocessed 

        return batch_images

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
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        # Formatting for Open-Flamingo desired input tokens
        prompts = [format_vqa_prompt(question, answer) for question, answer in zip(questions, answers)]        
        input_ids, attention_mask = prepare_text(prompts, max_length=self.context_len, tokenizer=self.tokenizer, device=self.device)
        
        images = self.load_images(images)

        labels = copy.deepcopy(input_ids)

        # Ignore the padding tokens, image token, and the end of chunk token for the loss
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        end_of_chunk_token_id = self.tokenizer.convert_tokens_to_ids(END_OF_CHUNK)
        labels[labels == image_token_id] = IGNORE_INDEX
        labels[labels == end_of_chunk_token_id] = IGNORE_INDEX

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        attention_mask = attention_mask[:, :self.tokenizer.model_max_length]

        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

        assert input_ids is not None, "input_ids should not be None for OpenFlamingo"
        assert attention_mask is not None, "attention_mask should not be None for OpenFlamingo"
        assert images is not None,"images should not be None for OpenFlamingo"
        assert labels is not None, "labels should not be None for loss computation"

        model_input_kwargs = {
            'vision_x': images,
            'lang_x': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'clear_conditioned_layers': True
        }

        outputs = self.model(**model_input_kwargs)

        # Shift so that tokens < n predict n
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        shift_labels = shift_labels.to(shift_logits.device)
        lm_prob = torch.zeros(shift_logits.shape[0])
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(shift_logits[k], shift_labels[k])).exp()
        return lm_prob