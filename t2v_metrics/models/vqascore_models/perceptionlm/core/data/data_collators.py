# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional

import torch

from core.distributed import get_is_master

logger = getLogger()


@dataclass
class MLLMBatch:
    x: torch.LongTensor
    y: torch.LongTensor
    mask: Optional[torch.BoolTensor] = None
    image_pos_index: Optional[torch.LongTensor] = None
    images: Optional[torch.Tensor] = None
    media_type: Optional[List[str]] = (["text"],)
    num_image_chunks: Optional[List[int]] = None

    def __post_init__(self):
        assert self.x.dim() == 2, "{} != 2".format(self.x.dim())
        assert self.x.shape == self.y.shape
        assert self.x.dtype == torch.int64
        assert self.y.dtype == torch.int64
        assert self.mask is None or self.mask.shape == self.x.shape


class BaseCollator:
    def __init__(
        self,
        tokenizer,
        show_first_batch: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.first_batch = show_first_batch

    def __call__(self, features: List[Dict[str, Any]]):
        raise NotImplementedError


class MllmPaddingCollator(BaseCollator):

    def prettify_decoded_text(self, texts: List[str]) -> List[str]:
        """
        Prettify the decoded text by replacing consecutive <|image|> tokens with a shortened form using regex.
        """
        prettified = []
        special_tokens = ["<|end_of_text|>", "<|image|>"]
        for text in texts:
            for token in special_tokens:
                # Regex to find consecutive occurrences of the token
                pattern = f"({re.escape(token)})\\1+"  # Captures repeating groups of the token

                def replace_consecutive(match):
                    count = len(match.group(0)) // len(token)
                    return f"{token}..x{count}"

                text = re.sub(pattern, replace_consecutive, text)
            prettified.append(text)
        return prettified

    def __call__(self, features: List[Dict[str, Any]]) -> MLLMBatch:
        text = []
        images = []
        media_type = []
        response_pos = []
        image_pos = []
        num_image_chunks = []
        for b in features:
            text.append(b["text_ids"])
            images.append(b["media"])
            response_pos.append(b["response_pos"])
            image_pos.append(b["image_pos"])
            num_image_chunks.append(b["num_image_chunks"])
            media_type.append(b["media_type"])

        images = [img for img in images if img is not None]
        images = torch.cat(images) if images else None

        # max_text_len = max([len(x) for x in text]) - 1
        bsz = len(text)
        input_ids = torch.full(
            (bsz, self.tokenizer.seq_len), self.tokenizer.pad_token_id
        )
        label_ids = torch.full(
            (bsz, self.tokenizer.seq_len), self.tokenizer.pad_token_id
        )
        image_pos_index = torch.full((bsz, self.tokenizer.seq_len), -1)

        for i in range(bsz):
            # Shift labels (list of lists) to train next token prediction
            for j in response_pos[i]:
                label_ids[i][j - 1] = text[i][j]
            # Remove last token for input
            text_len = len(text[i]) - 1
            input_ids[i][:text_len] = torch.tensor(text[i][:-1])
            # Fill image_pos_index
            if image_pos[i]:
                image_indices = torch.arange(len(image_pos[i]))
                image_pos_index[i, image_pos[i]] = image_indices

        mask = label_ids.ne(self.tokenizer.pad_token_id)

        # Replace all pad tokens with eos tokens
        input_ids[input_ids == self.tokenizer.pad_token_id] = (
            self.tokenizer.eos_token_id
        )
        label_ids[label_ids == self.tokenizer.pad_token_id] = (
            self.tokenizer.eos_token_id
        )

        if self.first_batch and get_is_master():
            input_decoded = self.tokenizer.decode_batch(input_ids)
            label_decoded = self.tokenizer.decode_batch(label_ids)
            logger.info(f"Input text: \n{self.prettify_decoded_text(input_decoded)}")
            logger.info(f"Label text: \n{self.prettify_decoded_text(label_decoded)}")
            self.first_batch = False

        return MLLMBatch(
            x=input_ids,
            y=label_ids,
            mask=mask,
            image_pos_index=image_pos_index,
            images=images,
            media_type=media_type,
            num_image_chunks=num_image_chunks,
        )
