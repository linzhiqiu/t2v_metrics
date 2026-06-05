import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Dict
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from .vqa_model import VQAScoreModel

QWEN3_VL_MODELS = {
    'qwen3-vl-235b-a22b': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-235B-A22B-Instruct'},
        'model': {
            'path': 'Qwen/Qwen3-VL-235B-A22B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-235b-a22b-thinking': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-235B-A22B-Thinking'},
        'model': {
            'path': 'Qwen/Qwen3-VL-235B-A22B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-30b-a3b': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-30B-A3B-Instruct'},
        'model': {
            'path': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-30b-a3b-thinking': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-30B-A3B-Thinking'},
        'model': {
            'path': 'Qwen/Qwen3-VL-30B-A3B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-32b': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-32B-Instruct'},
        'model': {
            'path': 'Qwen/Qwen3-VL-32B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-32b-thinking': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-32B-Thinking'},
        'model': {
            'path': 'Qwen/Qwen3-VL-32B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-8b': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-8B-Instruct'},
        'model': {
            'path': 'Qwen/Qwen3-VL-8B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-8b-thinking': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-8B-Thinking'},
        'model': {
            'path': 'Qwen/Qwen3-VL-8B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-4b': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-4B-Instruct'},
        'model': {
            'path': 'Qwen/Qwen3-VL-4B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-4b-thinking': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-4B-Thinking'},
        'model': {
            'path': 'Qwen/Qwen3-VL-4B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-2b': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-2B-Instruct'},
        'model': {
            'path': 'Qwen/Qwen3-VL-2B-Instruct',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3-vl-2b-thinking': {
        'tokenizer': {'path': 'Qwen/Qwen3-VL-2B-Thinking'},
        'model': {
            'path': 'Qwen/Qwen3-VL-2B-Thinking',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    # Qwen3.5 — same inference pattern as Qwen3-VL
    'qwen3.5-4b': {
        'tokenizer': {'path': 'Qwen/Qwen3.5-4B'},
        'model': {
            'path': 'Qwen/Qwen3.5-4B',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3.5-9b': {
        'tokenizer': {'path': 'Qwen/Qwen3.5-9B'},
        'model': {
            'path': 'Qwen/Qwen3.5-9B',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
    'qwen3.5-27b': {
        'tokenizer': {'path': 'Qwen/Qwen3.5-27B'},
        'model': {
            'path': 'Qwen/Qwen3.5-27B',
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
        'fps': 8.0,
    },
}


class Qwen3VLModel(VQAScoreModel):
    video_mode   = "direct"
    allows_image = True
    supports_trace = True

    def __init__(self,
                 model_name='qwen3-vl-235b-a22b',
                 device='cuda',
                 cache_dir=None,
                 checkpoint=None):
        assert model_name in QWEN3_VL_MODELS, \
            f"Model {model_name} not found in QWEN3_VL_MODELS"
        self.model_name  = model_name
        self.device      = device
        self.cache_dir   = cache_dir
        self.model_info  = QWEN3_VL_MODELS[model_name]
        self.checkpoint  = checkpoint if checkpoint else self.model_info['model']['path']
        self.load_model()

    def load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            torch_dtype=self.model_info['model']['torch_dtype'],
            attn_implementation=self.model_info['model']['attn_implementation'],
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_info['tokenizer']['path']
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def load_images(self, paths: List[str], fps: float = None) -> List[dict]:
        """Load images or videos and return them in Qwen3-VL format."""
        processed_data = []
        fps = fps if fps is not None else self.model_info.get('fps', 8.0)

        for path in paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                if fps == "dynamic":
                    processed_data.append({"type": "video", "video": path})
                else:
                    processed_data.append({"type": "video", "video": path, "fps": fps})
            elif path.lower().endswith('.npy'):
                np_array = np.load(path)
                if np_array.ndim == 3:
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    processed_data.append({"type": "image", "image": image})
                elif np_array.ndim == 4:
                    image = Image.fromarray(np_array[0].astype('uint8'), 'RGB')
                    processed_data.append({"type": "image", "image": image})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:
                image = Image.open(path).convert('RGB')
                processed_data.append({"type": "image", "image": image})

        return processed_data

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
    # Shared helper: build inputs via qwen-vl-utils + separate processor call
    # Bypasses transformers' video pipeline (which requires torchcodec/torchvision)
    # and routes through qwen-vl-utils with decord instead.
    # ------------------------------------------------------------------

    def _build_inputs(self, messages: List[dict]) -> dict:
        """
        Two-step input preparation using qwen-vl-utils:
        1. apply_chat_template(tokenize=False) — formats template only
        2. process_vision_info — decodes video/images via decord (qwen-vl-utils)
        3. processor call — tokenizes text + processes vision tensors

        This avoids transformers' built-in video pipeline which requires
        torchcodec or a newer torchvision, and uses Qwen's own preprocessing
        (smart_resize, temporal sampling, correct video tokens) instead.
        """
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **({'enable_thinking': False} if 'qwen3.5' in self.model_name else {})
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        # Qwen3-VL returns (video_tensor, video_metadata) tuples — unpack them
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs    = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None
        
        if 'fps' in video_kwargs and isinstance(video_kwargs['fps'], list):
            video_kwargs['fps'] = video_kwargs['fps'][0] if video_kwargs['fps'] else None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            do_resize=False,    # qwen-vl-utils already resized — avoid duplicate
            **(video_kwargs if video_inputs is not None else {})  # only pass for videos
        )
        return inputs.to(self.device)

    # ------------------------------------------------------------------
    # forward — VQAScore
    # ------------------------------------------------------------------

    def forward(self,
                images: List[str],
                texts: List[str],
                fps: float = None,
                question_template: str = 'Does this figure show "{}"? Please answer Yes or No.',
                answer_template: str = 'Yes',
                max_new_tokens: int = 1,
                temperature: float = 1.0,
                debug: bool = False) -> torch.Tensor:
        """
        Calculate alignment scores using the probability of the answer token(s).
        Temperature is applied manually to raw logits — HF receives temperature=1.0
        so it does not rescale internally.
        """
        assert len(images) == len(texts), "Number of images/videos and texts must match"

        questions      = [question_template.format(text) for text in texts]
        answers        = [answer_template.format(text)   for text in texts]
        processed_data = self.load_images(images, fps)

        lm_probs = []

        for idx, (data, question, answer) in enumerate(zip(processed_data, questions, answers)):
            if debug:
                print(f"\n{'='*60}")
                print(f"Sample {idx + 1}/{len(images)}")
                print(f"Path: {images[idx]}")
                print(f"Text: {texts[idx]}")

            messages = [
                {"role": "user", "content": [data, {"type": "text", "text": question}]}
            ]

            inputs = self._build_inputs(messages)

            answer_token_ids = self.processor.tokenizer.encode(
                answer, add_special_tokens=False
            )
            n_answer_tokens = len(answer_token_ids)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            generated_ids  = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            if debug:
                print(f"\nGenerated output:")
                print(f"  {generated_text}")

            last_token_id     = generated_ids[-1].item()
            special_token_ids = [
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.bos_token_id,
                self.processor.tokenizer.pad_token_id,
            ]

            offset = 0
            if last_token_id in special_token_ids:
                special_name = (
                    "EOS" if last_token_id == self.processor.tokenizer.eos_token_id else
                    "BOS" if last_token_id == self.processor.tokenizer.bos_token_id else "PAD"
                )
                if debug:
                    print(f"  Note: Last token is {special_name}, adjusting scoring")
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
                position                     = -(n_answer_tokens - i + offset)
                token_logits                 = outputs.scores[position][0]
                expected_token_id            = answer_token_ids[i]
                token_prob, token_probs_dist = self._compute_token_prob(
                    token_logits, expected_token_id, temperature
                )
                joint_prob *= token_prob

                if debug:
                    top_probs, top_indices = torch.topk(token_probs_dist, 5)
                    print(f"\n  Position {position} in outputs.scores:")
                    print(f"    Answer token: "
                          f"'{self.processor.tokenizer.decode([expected_token_id])}'  "
                          f"P={token_prob:.6f}")
                    print(f"    Top 5 alternatives:")
                    for rank, (p, tid) in enumerate(zip(top_probs, top_indices), 1):
                        tid_int     = tid.item()
                        is_expected = "✓" if tid_int == expected_token_id else " "
                        print(f"      {rank}. {is_expected} "
                              f"'{self.processor.tokenizer.decode([tid_int])}'  "
                              f"P={p.item():.6f}")

            geometric_mean_prob = joint_prob ** (1.0 / n_answer_tokens)

            if debug:
                print(f"\nJoint probability:          {joint_prob:.6f}")
                print(f"Geometric mean probability: {geometric_mean_prob:.6f}")

            lm_probs.append(geometric_mean_prob)

        if debug:
            print(f"\n{'='*60}")
            print(f"Final scores: {lm_probs}")

        return torch.tensor(lm_probs)

    # ------------------------------------------------------------------
    # forward_with_trace — VQAScore with debug output
    # ------------------------------------------------------------------

    def forward_with_trace(self,
                           images: List[str],
                           texts: List[str],
                           fps: float = None,
                           question_template: str = 'Does this figure show "{}"? Please answer Yes or No.',
                           answer_template: str = 'Yes',
                           max_new_tokens: int = 1,
                           temperature: float = 1.0,
                           score_position: str = "end",
                           debug: bool = False) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Calculate alignment scores with detailed trace information for debugging.
        Temperature is applied manually to raw logits — HF receives temperature=1.0
        so it does not rescale internally.

        Args:
            score_position: "end" scores the last n answer tokens (default),
                            "start" scores the first n answer tokens.
            debug:          If True, prints detailed token-level scoring information.
        """
        assert len(images) == len(texts), "Number of images/videos and texts must match"
        assert score_position in ("start", "end"), \
            f"score_position must be 'start' or 'end', got '{score_position}'"

        questions      = [question_template.format(text) for text in texts]
        answers        = [answer_template.format(text)   for text in texts]
        processed_data = self.load_images(images, fps)

        lm_probs = []
        traces   = []

        for idx, (data, question, answer) in enumerate(zip(processed_data, questions, answers)):
            if debug:
                print(f"\n{'='*60}")
                print(f"Sample {idx + 1}/{len(images)}")
                print(f"Path: {images[idx]}")
                print(f"Text: {texts[idx]}")

            messages = [
                {"role": "user", "content": [data, {"type": "text", "text": question}]}
            ]

            inputs = self._build_inputs(messages)

            answer_token_ids = self.processor.tokenizer.encode(
                answer, add_special_tokens=False
            )
            n_answer_tokens = len(answer_token_ids)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            generated_ids  = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            if debug:
                print(f"\nGenerated output:")
                print(f"  {generated_text}")

            if score_position == "start":
                score_start_idx = 0
                offset = 0
            else:  # "end"
                last_token_id     = generated_ids[-1].item()
                special_token_ids = [
                    self.processor.tokenizer.eos_token_id,
                    self.processor.tokenizer.bos_token_id,
                    self.processor.tokenizer.pad_token_id,
                ]
                if last_token_id in special_token_ids:
                    special_name = (
                        "EOS" if last_token_id == self.processor.tokenizer.eos_token_id else
                        "BOS" if last_token_id == self.processor.tokenizer.bos_token_id else "PAD"
                    )
                    if debug:
                        print(f"  Note: Last token is {special_name}, adjusting scoring")
                    n_answer_tokens = min(n_answer_tokens, len(outputs.scores) - 1)
                    offset = 1
                else:
                    offset = 0
                score_start_idx = len(generated_ids) - n_answer_tokens - offset

            if score_start_idx < 0:
                score_start_idx = 0

            available_tokens = len(outputs.scores) - score_start_idx
            if available_tokens < n_answer_tokens:
                print(f"  Warning: Only {available_tokens} tokens available at position, "
                      f"need {n_answer_tokens}, adjusting")
                n_answer_tokens  = available_tokens
                answer_token_ids = answer_token_ids[:n_answer_tokens]

            if n_answer_tokens <= 0:
                raise ValueError("No tokens available to score at the specified position")

            scored_indices     = list(range(score_start_idx, score_start_idx + n_answer_tokens))
            scored_token_ids   = generated_ids[
                score_start_idx:score_start_idx + n_answer_tokens
            ].tolist()
            scored_tokens_text = self.processor.tokenizer.decode(
                scored_token_ids, skip_special_tokens=True
            )

            if debug:
                print(f"\nScoring token(s): '{scored_tokens_text}'")
                print(f"  Token indices in generated sequence: {scored_indices}")

            joint_prob    = 1.0
            token_details = []

            for i in range(n_answer_tokens):
                score_idx                    = score_start_idx + i
                token_logits                 = outputs.scores[score_idx][0]
                expected_token_id            = answer_token_ids[i]
                token_prob, token_probs_dist = self._compute_token_prob(
                    token_logits, expected_token_id, temperature
                )
                joint_prob *= token_prob

                top_probs, top_indices = torch.topk(token_probs_dist, 5)
                alternatives = [
                    {
                        'token_id':    tid.item(),
                        'token_text':  self.processor.tokenizer.decode([tid.item()]),
                        'probability': p.item(),
                    }
                    for p, tid in zip(top_probs, top_indices)
                ]

                if debug:
                    print(f"\n  Position {score_idx} in outputs.scores "
                          f"(token index {scored_indices[i]} in sequence):")
                    print(f"    Answer Template token ID:   {expected_token_id}")
                    print(f"    Answer Template token text: "
                          f"'{self.processor.tokenizer.decode([expected_token_id])}'")
                    print(f"    P(answer_template): {token_prob:.6f}")
                    print(f"\n    Top 5 alternatives:")
                    for rank, alt in enumerate(alternatives, 1):
                        is_expected = "✓" if alt['token_id'] == expected_token_id else " "
                        print(f"      {rank}. ID={alt['token_id']:6d} | "
                              f"P={alt['probability']:.6f} | "
                              f"Text='{alt['token_text']}' {is_expected}")

                token_details.append({
                    'position':            score_idx,
                    'expected_token_id':   expected_token_id,
                    'expected_token_text': self.processor.tokenizer.decode([expected_token_id]),
                    'probability':         token_prob,
                    'top_alternatives':    alternatives,
                })

            geometric_mean_prob = joint_prob ** (1.0 / n_answer_tokens)

            if debug:
                print(f"\nJoint probability:          {joint_prob:.6f}")
                print(f"Geometric mean probability: {geometric_mean_prob:.6f}")

            traces.append({
                'generated_text':     generated_text,
                'generated_length':   len(generated_ids),
                'score_position':     score_position,
                'score_start_idx':    score_start_idx,
                'scored_indices':     scored_indices,
                'scored_tokens_text': scored_tokens_text,
                'probability':        geometric_mean_prob,
                'token_details':      token_details,
            })
            lm_probs.append(geometric_mean_prob)

        if debug:
            print(f"\n{'='*60}")
            print(f"Final scores: {lm_probs}")

        return torch.tensor(lm_probs), traces

    # ------------------------------------------------------------------
    # generate — free-form text generation
    # ------------------------------------------------------------------

    def generate(self,
                 images: List[str],
                 texts: List[str],
                 fps: float = None,
                 max_new_tokens: int = 2048,
                 temperature: float = 0.0,
                 do_sample: bool = None,
                 top_p: float = 0.9) -> List[str]:
        """
        Generate text responses for given images and text prompts.
        Note: temperature here controls HF sampling directly (not manually applied),
        since generation quality (not probability calibration) is the goal.
        """
        assert len(images) == len(texts), "Number of paths and texts must match"

        processed_data = self.load_images(images, fps)

        if do_sample is None:
            do_sample = (temperature > 0)

        generated_texts = []
        for data, text in zip(processed_data, texts):
            messages = [
                {"role": "user", "content": [data, {"type": "text", "text": text}]}
            ]

            inputs = self._build_inputs(messages)

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
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                generated_texts.append(output_text)

        return generated_texts