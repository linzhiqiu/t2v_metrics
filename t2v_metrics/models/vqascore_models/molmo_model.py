import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import List, Union

MOLMO_MODELS = {
    'molmo-72b-0924': {
        'model': {'path': 'allenai/Molmo-72B-0924'},
        'processor': {'path': 'allenai/Molmo-72B-0924'},
    },
    'molmo-7b-d-0924': {
        'model': {'path': 'allenai/Molmo-7B-D-0924'},
        'processor': {'path': 'allenai/Molmo-7B-D-0924'},
    },
    'molmo-7b-o-0924': {
        'model': {'path': 'allenai/Molmo-7B-O-0924'},
        'processor': {'path': 'allenai/Molmo-7B-O-0924'},
    },
    'molmoe-1b-0924': {
        'model': {'path': 'allenai/MolmoE-1B-0924'},
        'processor': {'path': 'allenai/MolmoE-1B-0924'},
    },
}

class MOLMOVisionModel:
    video_mode = "concat"
    allows_image = True
    def __init__(self,
                 model_name='molmo-7b-d-0924',
                 device='cuda',
                 cache_dir=None):
        assert model_name in MOLMO_MODELS, f"Model {model_name} not found in MOLMO_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = MOLMO_MODELS[model_name]
        self.load_model()

    def load_model(self):
        model_config = self.model_info['model']
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['path'],
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_info['processor']['path'],
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
            cache_dir=self.cache_dir
        )

    def load_images(self, paths: List[str]) -> List[Image.Image]:
        processed_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                raise NotImplementedError("Video processing is not supported for MOLMO Vision model.")
            else:  # Regular image file
                image = Image.open(path).convert('RGB')
                processed_data.append(image)
        return processed_data

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = "Does this image show \"{}\"? Answer the question with Yes or No",
                answer_template: str = "Yes") -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        images = self.load_images(paths)
        questions = [question_template.format(text) for text in texts]
        
        lm_probs = []
        for image, question in zip(images, questions):
            inputs = self.processor.process(images=[image], text=question)
            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                    ),
                    tokenizer=self.processor.tokenizer
                )
            
            # Get the logits for the first (and only) generated token
            logits = outputs.scores[0]
            
            # Get the index of the "Yes" token
            yes_token_id = self.processor.tokenizer.encode("Yes")[0]
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get the probability for the "Yes" token
            yes_prob = probs[0, yes_token_id].item()
            lm_probs.append(yes_prob)
        
        return torch.tensor(lm_probs)
    def generate(self,
                paths: List[str],
                texts: List[str],
                max_new_tokens: int = 256) -> List[str]:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        texts = [self.format_question(text) for text in texts]
        processed_data = self.load_images(paths)
        
        generated_texts = []
        for data, prompt in zip(processed_data, texts):
            if isinstance(data, torch.Tensor) and data.dim() == 4:  # Video
                image_sizes = [data.shape[2:] for _ in range(data.shape[0])]
                modalities = ["video"]
            else:  # Image
                image_sizes = [data.shape[1:]]
                modalities = None
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                images=[data],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                modalities=modalities
            )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(text.strip())

        return generated_texts

    def load_video(self, video_path, max_frames_num):
        raise NotImplementedError("Direct video processing is not supported for MOLMO Vision model.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)