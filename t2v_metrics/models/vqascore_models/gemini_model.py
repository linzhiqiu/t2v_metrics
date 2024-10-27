import base64
import time
from typing import List, Union
import torch
import google.generativeai as genai
from .vqa_model import VQAScoreModel

GEMINI_MODELS = {
    'gemini-1.5': {
        'model_path' : 'models/gemini-1.5-flash-002'
    },
}

def get_file_type(file_path):
    file_type = file_path.split('.')[-1].lower()
    if file_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']:
        return 'image'
    elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
        return 'video'
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

class GeminiModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True
    def __init__(self,
                 model_name='gemini-1.5',
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
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODELS[self.model_name]['model_path'])

    def load_images(self, paths: List[str]) -> List[dict]:
        loaded_data = []
        for path in paths:
            file_type = get_file_type(path)
            file = genai.upload_file(path=path)
            
            while file.state.name == "PROCESSING":
                time.sleep(3)
                file = genai.get_file(file.name)
            
            if file.state.name == "FAILED":
                raise ValueError(f"File processing failed: {path}")
            
            loaded_data.append({
                'path': path,
                'type': file_type,
                'file': file
            })
        return loaded_data

    def forward_single(self, data, question):
        try:
            response = self.model.generate_content(
                [question, data['file']],
                request_options={"timeout": 600},
                generation_config=genai.GenerationConfig(max_output_tokens=10, response_logprobs=True, candidate_count=2)
            )
            time.sleep(1.5) #This is added to prevent against quota limit of the free tier for gemini.
        
            candidates = response.candidates[0].logprobs_result.chosen_candidates
            
            for candidate in candidates:
                if candidate.token.lower() == 'yes':
                    return torch.tensor([candidate.log_probability]).exp()
                elif candidate.token.lower() == 'no':
                    return 1 - torch.tensor([candidate.log_probability]).exp()
            
            print(f"Warning: 'Yes' or 'No' not included in Gemini log probs: {data['path']} and question: {question}")
            return torch.tensor([0.0])

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
            lm_prob[idx] = self.forward_single(data, question)

        return lm_prob
    def generate_single(self, data, question):
        try:
            response = self.model.generate_content(
                [question, data['file']],
                request_options={"timeout": 600},
                generation_config=genai.GenerationConfig(max_output_tokens=256)
            )
            time.sleep(1.5)  # This is added to prevent against quota limit of the free tier for gemini.
            
            # Extract the generated text from the response
            generated_text = response.candidates[0].content.parts[0].text
            return generated_text

        except Exception as e:
            print(f"Failed file: {data['path']} and question: {question}")
            print(f"Error: {str(e)}")
            return ""

    def generate(self,
                paths: List[str],
                texts: List[str],
                num_frames: int = 5) -> List[str]:
        assert len(paths) == len(texts), "Number of paths and texts must match"
        
        questions = texts
        loaded_data = self.load_images(paths)

        generated_outputs = []

        for idx, (data, question) in enumerate(zip(loaded_data, questions)):
            generated_text = self.generate_single(data, question)
            generated_outputs.append(generated_text)

        return generated_outputs
        