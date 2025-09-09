# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import openai
import json
import time
from typing import List
import os

azure_endpoint = os.getenv("AZURE_ENDPOINT", "")
azure_api_key = os.getenv("OPENAI_API_KEY", "")

azure_gpt4v_client = openai.AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_version="2023-07-01-preview",
    api_key=azure_api_key,
    timeout = 120, 
)
azure_gpt4_client = openai.AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_version="2023-07-01-preview",
    api_key=azure_api_key,
    timeout = 120,
)
client = openai.AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_version="2023-07-01-preview",
    api_key=azure_api_key,
    timeout = 120,
)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def call_gemini_api(prompt: str, image_paths: List[str]=None, images_bs64: List[str]=None):
    assert image_paths is not None or images_bs64 is not None, "image_paths and images_bs64 cannot be both None."
    if images_bs64 is None:
        encoded_images = [encode_image(p) for p in image_paths]
    else:
        encoded_images = images_bs64
    completion = client.chat.completions.create(
        model="gemini-1.5-pro-preview", 
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    } for image in encoded_images],
                ]
            }
        ],
    )   
    return json.loads(completion.model_dump_json())['choices'][0]['message']['content']

def call_azure_gpt4v_api(prompt: str, image_paths: List[str]=None, images_bs64: List[str]=None):
    assert image_paths is not None or images_bs64 is not None, "image_paths and images_bs64 cannot be both None."
    if images_bs64 is None:
        encoded_images = [encode_image(p) for p in image_paths]
    else:
        encoded_images = images_bs64
    completion = azure_gpt4v_client.chat.completions.create(
        model="gptv", # or gpt-4o-2024-05-13
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": f'data:image/png;base64,{image}',
                            "detail": "high"
                        }
                    } for image in encoded_images],
                ]
            }
        ],
        temperature = 0.7,
        top_p = 0.95,
        # max_tokens = 4096,
    )
    return json.loads(completion.model_dump_json())['choices'][0]['message']['content']
        
def call_azure_gpt_api(prompt: str, model = 'gpt-35-turbo'):
    
    completion = azure_gpt4_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return json.loads(completion.model_dump_json())['choices'][0]['message']['content']

retry_exceptions = [
    "qpm limit, you can apply for expansion on the platform",
    "reach token limit, you can apply for expansion on the platform",
    "Request timed out",
    "The service is temporarily unable to process your request.",
    "upstream failed to respond",
    "502 Bad Gateway",
]

def try_call_api(model, prompt: str, image_paths: List[str]=None, images_bs64: List[str]=None):
    q_success = False
    while q_success != True:
        try:
            if model == 'gptv':
                assert image_paths is not None or images_bs64 is not None, "image_paths and images_bs64 cannot be both None."
                gpt_q = call_azure_gpt4v_api(prompt, image_paths, images_bs64)
            elif model in ['gpt-4-1106-preview', 'gpt-35-turbo']:
                gpt_q = call_azure_gpt_api(prompt, model)
            else:
                raise ValueError(f'{model} is invalid.')
            q_success = True
            return gpt_q, 0
        except Exception as e:
            e = f'ERROR from try_call_api: {e}'
            print(e)
            hit = False
            for x in retry_exceptions:
                if x in e:
                    hit = True
                    time.sleep(10)
            if not hit:
                return e, 1
 
if __name__ == '__main__':
    resp = call_azure_gpt4v_api(prompt='Describe the image in detail.', image_paths=[os.path.dirname(os.path.abspath(__file__)) + '/../assets/figures/tarsier_logo.jpg'])
