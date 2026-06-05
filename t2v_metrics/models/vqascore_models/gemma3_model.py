import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from .vqa_model import VQAScoreModel

GEMMA3_MODELS = {
    'gemma-3-4b-it': {
        'tokenizer': {'path': 'google/gemma-3-4b-it'},
        'model': {
            'path': 'google/gemma-3-4b-it',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
    },
    'gemma-3-12b-it': {
        'tokenizer': {'path': 'google/gemma-3-12b-it'},
        'model': {
            'path': 'google/gemma-3-12b-it',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
    },
    'gemma-3-27b-it': {
        'tokenizer': {'path': 'google/gemma-3-27b-it'},
        'model': {
            'path': 'google/gemma-3-27b-it',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
    },
}


class Gemma3Model(VQAScoreModel):
    video_mode   = "direct"
    allows_image = True
    allows_video = True  # via frame sampling

    def __init__(self,
                 model_name='gemma-3-12b-it',
                 device='cuda',
                 cache_dir=None,
                 checkpoint=None):
        assert model_name in GEMMA3_MODELS, \
            f"Model {model_name} not found in GEMMA3_MODELS"
        self.model_name = model_name
        self.device     = device
        self.cache_dir  = cache_dir
        self.model_info = GEMMA3_MODELS[model_name]
        self.checkpoint = checkpoint if checkpoint else self.model_info['model']['path']
        self.load_model()

    def load_model(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=self.model_info['model']['torch_dtype'],
            attn_implementation=self.model_info['model']['attn_implementation'],
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_info['tokenizer']['path']
        )
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(self,
                        video_path: str,
                        num_frames: int = 10) -> List[Tuple[float, Image.Image]]:
        """
        Extract num_frames evenly-spaced frames from a video.
        Returns list of (timestamp_seconds, PIL.Image) tuples.
        """
        vidcap       = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = vidcap.get(cv2.CAP_PROP_FPS)

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for i in frame_indices:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, image = vidcap.read()
            if success:
                image     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)
                timestamp = round(i / fps, 2)
                frames.append((timestamp, pil_image))
        vidcap.release()
        return frames

    # ------------------------------------------------------------------
    # load_images — returns list of content lists (one per sample)
    # ------------------------------------------------------------------

    def load_images(self,
                    paths: List[str],
                    num_frames: int = 10) -> List[List[dict]]:
        """
        Load images or videos and return as Gemma 3 message-content lists.
        Videos are sampled into frames with interleaved timestamps.
        """
        processed_data = []
        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                frames  = self._extract_frames(path, num_frames)
                content = []
                for timestamp, frame in frames:
                    content.append({"type": "text",  "text":  f"Frame at {timestamp}s:"})
                    content.append({"type": "image", "image": frame})
                processed_data.append(content)
            elif path.lower().endswith('.npy'):
                np_array = np.load(path)
                if np_array.ndim == 3:
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                elif np_array.ndim == 4:
                    image = Image.fromarray(np_array[0].astype('uint8'), 'RGB')
                else:
                    raise ValueError(f"Unexpected NumPy shape in {path}")
                processed_data.append([{"type": "image", "image": image}])
            else:
                image = Image.open(path).convert('RGB')
                processed_data.append([{"type": "image", "image": image}])
        return processed_data

    # ------------------------------------------------------------------
    # Helper: temperature-scaled softmax
    # ------------------------------------------------------------------

    def _compute_token_prob(self,
                            logits: torch.Tensor,
                            token_id: int,
                            temperature: float) -> Tuple[float, torch.Tensor]:
        token_probs_dist = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return token_probs_dist[token_id].item(), token_probs_dist

    # ------------------------------------------------------------------
    # Helper: build tokenized inputs from content list + question
    # ------------------------------------------------------------------

    def _build_inputs(self, content: List[dict], question: str):
        messages = [
            {
                "role": "user",
                "content": content + [{"type": "text", "text": question}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.model.dtype)
        return inputs

    # ------------------------------------------------------------------
    # forward — VQAScore
    # ------------------------------------------------------------------

    def forward(self,
                images: List[str],
                texts: List[str],
                num_frames: int = 10,
                fps=None,
                question_template: str = 'Does this figure show "{}"? Please answer Yes or No.',
                answer_template: str = 'Yes',
                max_new_tokens: int = 1,
                temperature: float = 1.0) -> torch.Tensor:
        assert len(images) == len(texts), "Number of images and texts must match"

        questions      = [question_template.format(t) for t in texts]
        answers        = [answer_template.format(t)   for t in texts]
        processed_data = self.load_images(images, num_frames)

        lm_probs = []
        for content, question, answer in zip(processed_data, questions, answers):
            inputs = self._build_inputs(content, question)

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

            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]

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
                 num_frames: int = 10,
                 fps=None,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.0,
                 do_sample: bool = None,
                 top_p: float = 0.9) -> List[str]:
        """
        Generate text responses for given images/videos and prompts.
        Videos are sampled into num_frames frames with interleaved timestamps.
        """
        assert len(images) == len(texts), "Number of images and texts must match"

        processed_data = self.load_images(images, num_frames)

        if do_sample is None:
            do_sample = (temperature > 0)

        generated_texts = []
        for content, text in zip(processed_data, texts):
            inputs = self._build_inputs(content, text)
            in_len = inputs["input_ids"].shape[-1]

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
                output_ids = self.model.generate(**inputs, **generation_kwargs)

            response = self.processor.decode(
                output_ids[0][in_len:], skip_special_tokens=True
            ).strip()
            generated_texts.append(response)

        return generated_texts