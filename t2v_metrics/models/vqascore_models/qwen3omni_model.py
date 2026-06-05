import torch
import numpy as np
import soundfile as sf
from PIL import Image
from typing import List, Union, Tuple, Dict, Optional
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from .vqa_model import VQAScoreModel

import av

QWEN3_OMNI_MODELS = {
    'qwen3-omni-30b-a3b-captioner': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Captioner',
        },
        'model': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Captioner',
            'dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
    },
    'qwen3-omni-30b-a3b': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
            'dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
    },
    'qwen3-omni-30b-a3b-thinking': {
        'tokenizer': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Thinking',
        },
        'model': {
            'path': 'Qwen/Qwen3-Omni-30B-A3B-Thinking',
            'dtype': torch.bfloat16,
            'attn_implementation': 'sdpa',
        },
    },
}



class Qwen3OmniModel(VQAScoreModel):
    video_mode  = "direct"
    allows_image = True
    allows_audio = True
    supports_trace = True
    def __init__(self,
                 model_name='qwen3-omni-30b-a3b',
                 device='cuda',
                 cache_dir=None,
                 checkpoint=None,
                 use_audio_in_video=True):
        assert model_name in QWEN3_OMNI_MODELS, \
            f"Model {model_name} not found in QWEN3_OMNI_MODELS"
        self.model_name         = model_name
        self.device             = device
        self.cache_dir          = cache_dir
        self.model_info         = QWEN3_OMNI_MODELS[model_name]
        self.checkpoint         = checkpoint if checkpoint else self.model_info['model']['path']
        self.use_audio_in_video = use_audio_in_video
        self.load_model()

    def load_model(self):
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.checkpoint,
            dtype=self.model_info['model']['dtype'],
            attn_implementation=self.model_info['model']['attn_implementation'],
            device_map="auto"
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.model_info['tokenizer']['path']
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # load_images — satisfies abstract base class + handles audio
    # ------------------------------------------------------------------

    def load_images(self,
                    paths: List[str],
                    audio_paths: Optional[List[str]] = None) -> List[List[dict]]:
        """
        Load images, videos, and optionally audio files.
        Returns a list of content lists (one per sample) in Qwen3-Omni format.
        """
        processed_data = []

        for i, path in enumerate(paths):
            content = []

            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                content.append({"type": "video", "video": path})
            elif path.lower().endswith('.npy'):
                np_array = np.load(path)
                if np_array.ndim == 3:
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    content.append({"type": "image", "image": image})
                elif np_array.ndim == 4:
                    image = Image.fromarray(np_array[0].astype('uint8'), 'RGB')
                    content.append({"type": "image", "image": image})
                else:
                    raise ValueError(f"Unexpected NumPy shape in {path}")
            else:
                image = Image.open(path).convert('RGB')
                content.append({"type": "image", "image": image})

            if audio_paths and i < len(audio_paths) and audio_paths[i]:
                content.append({"type": "audio", "audio": audio_paths[i]})

            processed_data.append(content)

        return processed_data

    # ------------------------------------------------------------------
    # Shared helper: temperature-scaled softmax (consistent with other models)
    # ------------------------------------------------------------------

    def _compute_token_prob(self,
                            logits: torch.Tensor,
                            token_id: int,
                            temperature: float) -> Tuple[float, torch.Tensor]:
        """
        Apply temperature manually to raw logits before softmax.
        HF always receives temperature=1.0 so logits stay unscaled during
        generation; we own the scaling here instead.
        """
        token_probs_dist = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return token_probs_dist[token_id].item(), token_probs_dist

    # ------------------------------------------------------------------
    # Shared helper: prepare inputs for a single conversation turn
    # ------------------------------------------------------------------

    def _video_has_audio(self, video_path: str) -> bool:
        """Check if a video file has an audio track."""
        try:
            container = av.open(video_path)
            has_audio = len(container.streams.audio) > 0
            container.close()
            return has_audio
        except Exception:
            return False

    def _prepare_inputs(self, content: List[dict], question: str):
        has_video = any(c.get("type") == "video" for c in content)
        if has_video and self.use_audio_in_video:
            video_path = next(c["video"] for c in content if c.get("type") == "video")
            effective_audio = self._video_has_audio(video_path)
        else:
            effective_audio = self.use_audio_in_video

        conversation = [
            {"role": "user", "content": content + [{"type": "text", "text": question}]}
        ]
        text   = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=effective_audio
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=effective_audio,
        )
        inputs = inputs.to(self.device)
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.model.dtype)

        return inputs, effective_audio  # return both

    # ------------------------------------------------------------------
    # forward — VQAScore
    # ------------------------------------------------------------------
    def forward(self,
                paths: List[str],
                texts: List[str],
                audio_paths: Optional[List[str]] = None,
                fps=None,
                question_template: str = 'Does this figure show "{}"? Please answer Yes or No.',
                answer_template: str = 'Yes',
                max_new_tokens: int = 1,
                temperature: float = 1.0,
                debug: bool = False) -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions      = [question_template.format(t) for t in texts]
        answers        = [answer_template.format(t)   for t in texts]
        processed_data = self.load_images(paths, audio_paths)

        lm_probs = []
        for idx, (content, question, answer) in enumerate(zip(processed_data, questions, answers)):
            if debug:
                print(f"\n{'='*60}")
                print(f"Sample {idx + 1}/{len(paths)}")
                print(f"Path: {paths[idx]}")
                print(f"Text: {texts[idx]}")

            inputs, effective_audio = self._prepare_inputs(content, question)

            answer_token_ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            n_answer_tokens  = len(answer_token_ids)

            with torch.inference_mode():
                result = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=effective_audio,
                )
            outputs = result[0] if isinstance(result, tuple) else result

            generated_ids  = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

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
                if debug:
                    special_name = (
                        "EOS" if last_token_id == self.processor.tokenizer.eos_token_id else
                        "BOS" if last_token_id == self.processor.tokenizer.bos_token_id else "PAD"
                    )
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
                expected_id                  = answer_token_ids[i]
                token_prob, token_probs_dist = self._compute_token_prob(
                    token_logits, expected_id, temperature
                )
                joint_prob *= token_prob

                if debug:
                    top_probs, top_indices = torch.topk(token_probs_dist, 5)
                    print(f"\n  Position {position} in outputs.scores:")
                    print(f"    Answer token: '{self.processor.tokenizer.decode([expected_id])}'  "
                        f"P={token_prob:.6f}")
                    print(f"    Top 5 alternatives:")
                    for rank, (p, tid) in enumerate(zip(top_probs, top_indices), 1):
                        tid_int     = tid.item()
                        is_expected = "✓" if tid_int == expected_id else " "
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


    def forward_with_trace(self,
                        paths: List[str],
                        texts: List[str],
                        audio_paths: Optional[List[str]] = None,
                        fps=None,
                        question_template: str = 'Does this figure show "{}"? Please answer Yes or No.',
                        answer_template: str = 'Yes',
                        max_new_tokens: int = 1,
                        temperature: float = 1.0,
                        score_position: str = "end",
                        debug: bool = False) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Calculate alignment scores with detailed trace information.
        Temperature is applied manually to raw logits — HF receives temperature=1.0.

        Args:
            score_position: "end" scores the last n answer tokens (default),
                            "start" scores the first n answer tokens.
            debug:          If True, prints detailed token-level scoring information.
        """
        assert len(paths) == len(texts), "Number of paths and texts must match"
        assert score_position in ("start", "end"), \
            f"score_position must be 'start' or 'end', got '{score_position}'"

        questions      = [question_template.format(t) for t in texts]
        answers        = [answer_template.format(t)   for t in texts]
        processed_data = self.load_images(paths, audio_paths)

        lm_probs = []
        traces   = []

        for idx, (content, question, answer) in enumerate(
            zip(processed_data, questions, answers)
        ):
            if debug:
                print(f"\n{'='*60}")
                print(f"Sample {idx + 1}/{len(paths)}")
                print(f"Path: {paths[idx]}")
                print(f"Text: {texts[idx]}")

            inputs, effective_audio = self._prepare_inputs(content, question)  # fix: unpack tuple

            answer_token_ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            n_answer_tokens  = len(answer_token_ids)

            with torch.inference_mode():
                result = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=effective_audio,  # now defined
                )
            outputs = result[0] if isinstance(result, tuple) else result

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
            scored_token_ids   = generated_ids[score_start_idx:score_start_idx + n_answer_tokens].tolist()
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
    # generate — free-form text (and optionally audio) generation
    # ------------------------------------------------------------------
    def generate(self,
                images: List[str],
                texts: List[str],
                audio_paths: Optional[List[str]] = None,
                fps=None,
                max_new_tokens: int = 256,
                return_audio: bool = False,
                speaker: str = "Ethan",
                save_audio_path: Optional[str] = None) -> Union[List[str], Tuple[List[str], list]]:
        """
        Generate text (and optionally audio) responses.

        Args:
            return_audio: If True, also generate and return audio output.
                        Defaults to False — text-only for general use.

        Returns:
            List of text responses, or (texts, audios) tuple if return_audio=True.
        """
        assert len(images) == len(texts), "Number of paths and texts must match"

        processed_data   = self.load_images(images, audio_paths)
        generated_texts  = []
        generated_audios = []

        for content, text in zip(processed_data, texts):
            inputs, effective_audio = self._prepare_inputs(content, text)

            with torch.inference_mode():
                if return_audio:
                    text_ids, audio = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        speaker=speaker,
                        thinker_return_dict_in_generate=True,
                        use_audio_in_video=effective_audio,
                    )
                else:
                    # Text-only — skip audio computation entirely
                    result = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        thinker_return_dict_in_generate=True,
                        use_audio_in_video=effective_audio,
                    )
                    text_ids = result[0] if isinstance(result, tuple) else result
                    audio    = None

            output_text = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            generated_texts.append(output_text)

            if audio is not None:
                audio_data = audio.reshape(-1).detach().cpu().numpy()
                generated_audios.append(audio_data)
                if save_audio_path:
                    sf.write(save_audio_path, audio_data, samplerate=24000)
            else:
                generated_audios.append(None)

        if return_audio and any(a is not None for a in generated_audios):
            return generated_texts, generated_audios
        return generated_texts