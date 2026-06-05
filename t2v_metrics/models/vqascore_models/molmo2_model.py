import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from transformers import AutoModelForImageTextToText, AutoProcessor
from .vqa_model import VQAScoreModel

MOLMO2_MODELS = {
    'molmo2-4b': {
        'tokenizer': {
            'path': 'allenai/Molmo2-4B',
            'trust_remote_code': True,
        },
        'model': {
            'path': 'allenai/Molmo2-4B',
            'trust_remote_code': True,
        },
    },
    'molmo2-7b': {
        'tokenizer': {
            'path': 'allenai/Molmo2-O-7B',
            'trust_remote_code': True,
        },
        'model': {
            'path': 'allenai/Molmo2-O-7B',
            'trust_remote_code': True,
        },
    },
    'molmo2-8b': {
        'tokenizer': {
            'path': 'allenai/Molmo2-8B',
            'trust_remote_code': True,
        },
        'model': {
            'path': 'allenai/Molmo2-8B',
            'trust_remote_code': True,
        },
    },
}


class Molmo2Model(VQAScoreModel):
    video_mode   = "direct"
    allows_image = True
    allows_video = True  # native video support — no frame extraction needed

    def __init__(self,
                 model_name='molmo2-8b',
                 device='cuda',
                 cache_dir=None,
                 checkpoint=None):
        assert model_name in MOLMO2_MODELS, \
            f"Model {model_name} not found in MOLMO2_MODELS"
        self.model_name = model_name
        self.device     = device
        self.cache_dir  = cache_dir
        self.model_info = MOLMO2_MODELS[model_name]
        self.checkpoint = checkpoint if checkpoint else self.model_info['model']['path']
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_info['tokenizer']['path'],
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        ).eval()
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # load_images — returns one media dict per sample
    # ------------------------------------------------------------------

    def load_images(self, paths: List[str]) -> List[dict]:
        """
        Load images or videos and return in Molmo2 content-dict format.
        Videos are passed as paths — the model handles frame sampling internally.
        """
        processed_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                processed_data.append({"type": "video", "video": path})
            elif path.lower().endswith('.npy'):
                np_array = np.load(path)
                if np_array.ndim == 3:
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                elif np_array.ndim == 4:
                    image = Image.fromarray(np_array[0].astype('uint8'), 'RGB')
                else:
                    raise ValueError(f"Unexpected NumPy shape in {path}")
                processed_data.append({"type": "image", "image": image})
            else:
                image = Image.open(path).convert('RGB')
                processed_data.append({"type": "image", "image": image})
        return processed_data

    # ------------------------------------------------------------------
    # Helper: temperature-scaled softmax
    # ------------------------------------------------------------------

    def _compute_token_prob(self,
                            logits: torch.Tensor,
                            token_id: int,
                            temperature: float) -> Tuple[float, torch.Tensor]:
        """
        Apply temperature manually to raw logits before softmax.
        HF always receives temperature=1.0 so logits stay unscaled during
        generation — we own the scaling here instead.
        """
        token_probs_dist = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return token_probs_dist[token_id].item(), token_probs_dist

    # ------------------------------------------------------------------
    # Helper: build tokenized inputs
    # ------------------------------------------------------------------

    def _build_inputs(self, media: dict, question: str):
        messages = [
            {
                "role": "user",
                "content": [
                    media,
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # forward — VQAScore
    # ------------------------------------------------------------------

    def forward(self,
                images: List[str],
                texts: List[str],
                fps=None,
                question_template: str = 'Does this figure show "{}"? Please answer Yes or No.',
                answer_template: str = 'Yes',
                max_new_tokens: int = 1,
                temperature: float = 1.0) -> torch.Tensor:
        assert len(images) == len(texts), "Number of images and texts must match"

        questions      = [question_template.format(t) for t in texts]
        answers        = [answer_template.format(t)   for t in texts]
        processed_data = self.load_images(images)

        lm_probs = []
        for media, question, answer in zip(processed_data, questions, answers):
            inputs = self._build_inputs(media, question)

            answer_token_ids = self.processor.tokenizer.encode(
                answer, add_special_tokens=False
            )
            n_answer_tokens = len(answer_token_ids)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,   # HF must not apply temperature — we do it manually
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]

            last_token_id     = generated_ids[-1].item()
            special_token_ids = [
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.bos_token_id,
                self.processor.tokenizer.pad_token_id,
            ]

            offset = 0
            if last_token_id in special_token_ids:
                n_answer_tokens = min(n_answer_tokens, len(outputs.scores) - 1)
                offset = 1
                if n_answer_tokens <= 0:
                    raise ValueError("No content tokens to score after removing special tokens")

            if len(outputs.scores) < n_answer_tokens:
                print(f"  Warning: Generated {len(outputs.scores)} tokens but need "
                      f"{n_answer_tokens}, adjusting")
                n_answer_tokens  = len(outputs.scores)
                answer_token_ids = answer_token_ids[:n_answer_tokens]

            joint_prob = 1.0
            for i in range(n_answer_tokens):
                position      = -(n_answer_tokens - i + offset)
                token_logits  = outputs.scores[position][0]
                expected_id   = answer_token_ids[i]
                token_prob, _ = self._compute_token_prob(
                    token_logits, expected_id, temperature
                )
                joint_prob *= token_prob

            geometric_mean_prob = joint_prob ** (1.0 / n_answer_tokens)
            lm_probs.append(geometric_mean_prob)

        return torch.tensor(lm_probs)

    # ------------------------------------------------------------------
    # generate — free-form text generation
    # ------------------------------------------------------------------

    def generate(self,
                 images: List[str],
                 texts: List[str],
                 fps=None,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.0,
                 do_sample: bool = None,
                 top_p: float = 0.9) -> List[str]:
        """
        Generate text responses for given images/videos and prompts.
        Videos are handled natively — pass the path directly.
        """
        assert len(images) == len(texts), "Number of images and texts must match"

        processed_data = self.load_images(images)

        if do_sample is None:
            do_sample = (temperature > 0)

        generated_texts = []
        for media, text in zip(processed_data, texts):
            inputs = self._build_inputs(media, text)
            in_len = inputs['input_ids'].shape[1]

            generation_kwargs = {"max_new_tokens": max_new_tokens}
            if do_sample and temperature > 0:
                generation_kwargs.update({
                    "do_sample":   True,
                    "temperature": temperature,
                    "top_p":       top_p,
                })
            else:
                generation_kwargs["do_sample"] = False

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)

            generated_tokens = generated_ids[0, in_len:]
            response = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            generated_texts.append(response)

        return generated_texts