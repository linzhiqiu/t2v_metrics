import torch
import numpy as np
import os
import tempfile
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
from PIL import Image

from .vqa_model import VQAScoreModel

INTERNLMXCOMPOSER25_MODELS = {
    'internlmxcomposer25-7b': {
        'tokenizer': {
            'path': 'internlm/internlm-xcomposer2d5-7b',
            'trust_remote_code': True,
        },
        'model': {
            'pretrained_model_name_or_path': 'internlm/internlm-xcomposer2d5-7b',
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
        },
    },

}

class InternLMXComposer25Model(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='internlmxcomposer25-7b',
                 device='cuda',
                 cache_dir=None):
        assert model_name in INTERNLMXCOMPOSER25_MODELS, f"Model {model_name} not found in INTERNLMXCOMPOSER25_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = INTERNLMXCOMPOSER25_MODELS[model_name]
        self.load_model()
        

    def load_model(self):
        tokenizer_path = self.model_info['tokenizer']['path']
        
        self.model = AutoModel.from_pretrained(
            **self.model_info['model']
        ).eval().to(self.device).half()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **self.model_info['tokenizer']
        )
        self.model.tokenizer = self.tokenizer

    def process_path(self, path: str) -> str:
        if path.lower().endswith('.npy'):
            # Create a temporary file with .jpg extension
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
            
            # Load the .npy file and save it as a temporary .jpg file
            np_array = np.load(path)
            if np_array.ndim == 3 and np_array.shape[2] == 3:  # Ensure it's an RGB image
                from PIL import Image
                Image.fromarray(np_array.astype('uint8'), 'RGB').save(temp_path)
            else:
                raise ValueError(f"Unexpected shape for NumPy array in {path}")
            
            return temp_path
        else:
            return path

    def load_images(self, paths: List[str]) -> List[str]:
        processed_paths = []
        for path in paths:
            if path.startswith(("http://", "https://")):
                raise NotImplementedError("Web link image/video inputs are not yet supported for this model. Please use a local path, or otherwise, make a Github issue request if this feature is necessary.")
            processed_paths.append(self.process_path(path))
        return processed_paths

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = "Does this figure show \"{}\"? Please answer yes or no.",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_paths = self.load_images(paths)

        lm_probs = []
        temp_files = []
        for path, question, answer in zip(processed_paths, questions, answers):
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
                query = f"Here are some frames of a video. {question}"
            else:  # Image file
                query = f"<ImageHere> {question}"
            
            use_meta = True
            image = [path]
            history = []
            meta_instruction = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
            '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
            '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
            '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.'
            streamer = None
            num_beams = 1
            do_sample=False
            infer_mode='base'
            hd_num = 24

            # Ensure all input images are in RGB:
            # with Image.open(path) as img:
            #     # If the image has an alpha channel (RGBA), convert it to RGB
            #     if img.mode == 'RGBA':
            #         img = img.convert('RGB')
            #         # Overwrite the original image or save to a new directory
            #         img.save(path)  # Overwrite the original file
            #     elif img.mode == 'LA':  # Convert LA (grayscale with alpha) to RGB
            #         img = img.convert('RGB')
            #         img.save(path)
        
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if not use_meta:
                    meta_instruction = ''
                if image is None:
                    inputs = self.model.build_inputs(self.tokenizer, query, history, meta_instruction)
                    im_mask = torch.zeros(inputs['input_ids'].shape[:2]).cuda().bool()
                else:
                    inputs, im_mask, _ = self.model.interleav_wrap_chat(query, image, history=history, meta_instruction=meta_instruction, hd_num=hd_num)
                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items() if torch.is_tensor(v)
                }
                # also add end-of-assistant token in eos token id to avoid unnecessary generation
                eos_token_id = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
                ]
                outputs = self.model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=1,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    eos_token_id=eos_token_id,
                    repetition_penalty=1.005,
                    im_mask=im_mask,
                    infer_mode='base',
                    output_scores=True, 
                    return_dict_in_generate=True
                )
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            # print(f'Yes Tokens {self.tokenizer.encode("Yes")}') InternLM adds a beginning of sentence token
            yes_token_id = self.tokenizer.encode(answer)[1]
            lm_prob = probs[0, yes_token_id].item()
            lm_probs.append(lm_prob)

        
            # If this is a temporary file, add it to the list for cleanup
            if path.startswith(tempfile.gettempdir()):
                temp_files.append(path)

        # Clean up temporary files
        for temp_file in temp_files:
            os.remove(temp_file)

        return torch.tensor(lm_probs)
    def generate(self,
            images: List[str],
            texts: List[str],
            max_new_tokens: int = 256) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"

        questions = texts
        processed_paths = self.load_images(images)

        generated_outputs = []
        temp_files = []
        for path, question in zip(processed_paths, questions):
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                query = f"Here are some frames of a video. {question}"
            else:
                query = f"<ImageHere> {question}"
            
            use_meta = True
            image = [path]
            history = []
            meta_instruction = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
            '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
            '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
            '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.'
            streamer = None
            num_beams = 5
            do_sample = False
            infer_mode = 'base'
            hd_num = 24

            # with Image.open(path) as img:
            #     if img.mode == 'RGBA':
            #         img = img.convert('RGB')
            #         img.save(path)
            #     elif img.mode == 'LA':
            #         img = img.convert('RGB')
            #         img.save(path)
        
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if not use_meta:
                    meta_instruction = ''
                if image is None:
                    inputs = self.model.build_inputs(self.tokenizer, query, history, meta_instruction)
                    im_mask = torch.zeros(inputs['input_ids'].shape[:2]).cuda().bool()
                else:
                    inputs, im_mask, _ = self.model.interleav_wrap_chat(query, image, history=history, meta_instruction=meta_instruction, hd_num=hd_num)
                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items() if torch.is_tensor(v)
                }
                eos_token_id = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
                ]
                outputs = self.model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    eos_token_id=eos_token_id,
                    repetition_penalty=1.005,
                    im_mask=im_mask,
                    infer_mode='base'
                )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                generated_outputs.append(generated_text)

            if path.startswith(tempfile.gettempdir()):
                temp_files.append(path)

        for temp_file in temp_files:
            os.remove(temp_file)

        return generated_outputs