import time
from typing import List
import torch
from google import genai
from google.genai.types import HttpOptions, Part, GenerateContentConfig
from .vqa_model import VQAScoreModel

GEMINI_MODELS = {
    'gemini-1.5-flash': {
        'model_path': 'gemini-1.5-flash-002'
    },
    'gemini-2.0': {
        'model_path': 'gemini-2.0-flash-001'
    },
    'gemini-2.0-pro': {
        'model_path': 'gemini-2.0-pro-exp-02-05'
    }
}

def get_file_type(file_path):
    file_type = file_path.split('.')[-1].lower()
    if file_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']:
        return 'image'
    elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
        return 'video'
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def get_mime_type(file_path):
    file_type = get_file_type(file_path)
    extension = file_path.split('.')[-1].lower()
    
    if file_type == 'image':
        return f"image/{extension if extension != 'jpg' else 'jpeg'}"
    elif file_type == 'video':
        return f"video/{extension}"
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

class GeminiModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    
    def __init__(self,
                 model_name='gemini-1.5-flash',
                 device='cuda',
                 cache_dir=None,
                 api_key=None,
                 top_logprobs=2):
        assert model_name in GEMINI_MODELS
        assert api_key is not None, "Please provide a Google API key"
        self.api_key = api_key
        self.top_logprobs = top_logprobs

        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=HttpOptions(api_version="v1")
        )
        self.model_path = GEMINI_MODELS[self.model_name]['model_path']

    def load_images(self, paths: List[str]) -> List[dict]:
        loaded_data = []
        for path in paths:
            file_type = get_file_type(path)
            mime_type = get_mime_type(path)
            
            # Read file content
            with open(path, 'rb') as f:
                file_content = f.read()
            
            loaded_data.append({
                'path': path,
                'type': file_type,
                'mime_type': mime_type,
                'content': file_content
            })
        return loaded_data
    
    def forward_single(self, data, question, answer):
        try:
            response = self.client.models.generate_content(
                model=self.model_path,
                contents=[
                    question,
                    Part.from_bytes(data=data['content'], mime_type=data['mime_type'])
                ],
                config=GenerateContentConfig(
                    max_output_tokens=64
                )
            )

            time.sleep(1.5)  # This is added to prevent quota limit of the free tier for gemini.
        
            # Extract the generated text
            generated_text = response.text.strip().lower()
            
            # Simple string matching
            if answer.lower() == "yes":
                # If "yes" is in the generated text, return 1.0, otherwise 0.0
                score = 1.0 if "yes" in generated_text else 0.0
            else:
                # If looking for a different answer, check if that answer is in the text
                score = 1.0 if answer.lower() in generated_text else 0.0
                
            return torch.tensor([score])

        except Exception as e:
            print(f"Failed file: {data['path']} and question: {question}")
            print(f"Error: {str(e)}")
            return torch.tensor([0.0])

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str ='Does this image show "{}"? Please answer yes or no.',
                answer_template: str = 'Yes',
                num_frames: int = 5) -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = [question_template.format(text) for text in texts]
        loaded_data = self.load_images(paths)

        lm_prob = torch.zeros(len(paths))

        for idx, (data, question) in enumerate(zip(loaded_data, questions)):
            lm_prob[idx] = self.forward_single(data, question, answer_template)

        return lm_prob
        
    def generate_single(self, data, question):
        try:
            response = self.client.models.generate_content(
                model=self.model_path,
                contents=[
                    question,
                    Part.from_bytes(data=data['content'], mime_type=data['mime_type'])
                ],
                config=GenerateContentConfig(
                    max_output_tokens=256
                )
            )
      
            time.sleep(1.5)  # This is added to prevent quota limit of the free tier for gemini.
            
            # Extract the generated text from the response
            generated_text = response.text
            return generated_text

        except Exception as e:
            print(f"Failed file: {data['path']} and question: {question}")
            print(f"Error: {str(e)}")
            return ""

    def generate(self,
                images: List[str],
                texts: List[str],
                num_frames: int = 5) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        questions = texts
        loaded_data = self.load_images(images)

        generated_outputs = []

        for idx, (data, question) in enumerate(zip(loaded_data, questions)):
            generated_text = self.generate_single(data, question)
            generated_outputs.append(generated_text)

        return generated_outputs

         