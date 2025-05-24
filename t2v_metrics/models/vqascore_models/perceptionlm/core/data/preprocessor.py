# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from apps.plm.dataset_conf import DatasetConf

logger = logging.getLogger(__name__)


class VisionPreprocessor:
    def __init__(
        self,
        transform: Optional[Callable],
        tokenizer: Callable,
        max_video_frames: Optional[int],
        dataset_config: DatasetConf,
    ):
        self.mllm_tokenizer = tokenizer
        self.transform = transform
        self.root_dir = ""
        if dataset_config.root_dir:
            self.root_dir = dataset_config.root_dir
        self.max_video_frames = max_video_frames

    def __call__(self, row: Dict[str, Any], rng: np.random.RandomState):
        try:
            return self.process(row, rng)
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            return None  # None will be skipped in training

    def process(self, row: Dict[str, Any], rng: np.random.RandomState):
        del rng
        if "conversations" in row:
            conversations = row["conversations"]
        else:
            conversations = self.get_conversation(caption=row["text"], prompt="")

        if "bbox" in row:
            assert (
                "width" in row and "height" in row
            ), f"bbox is present in the annotation, however image width or height is not specified, which is not expected."
            w, h = row["width"], row["height"]
            bboxes = row["bbox"]
            conversations = self.transform["region"](conversations, bboxes, w, h)

        media = None
        media_type = ""
        if "image" in row:
            processed_images = []
            image_files = row["image"]
            if isinstance(image_files, str):
                image_files = [image_files]
            pil_images = []
            for image_file in image_files:
                if self.root_dir:
                    image_file = os.path.join(self.root_dir, image_file)
                try:
                    image = Image.open(image_file).convert("RGB")
                    pil_images.append(image)
                except Exception as e:
                    logger.info(
                        f"loading image failed because of the following error:\n {e}"
                    )
                    return None
            if self.transform:
                if len(pil_images) == 1:
                    transform = self.transform["image"]
                    processed_images, _ = transform(pil_images[0])
                else:
                    transform = self.transform["video"]
                    processed_images, _ = transform._process_multiple_images_pil(
                        pil_images
                    )
            if len(processed_images.shape) == 3:
                processed_images = processed_images.unsqueeze(0)
            media = processed_images
            media_type = "multi_image" if len(image_files) > 1 else "image"
        elif "video" in row:
            video_file = row["video"]
            start_time = row.get("start_time", None)
            bbox_map = row.get("bbox_map", None)
            end_time = row.get("end_time", None)
            if self.root_dir:
                video_file = os.path.join(self.root_dir, video_file)
            video_info = (
                video_file,
                self.max_video_frames,
                start_time,
                end_time,
                bbox_map,
            )
            video, _ = self.transform["video"](video_info)
            media = video
            media_type = "video"
        else:
            # This is a text-only sample. We create a dummy white image to facilitate batch processing.
            # Note that this image serves solely as a placeholder and is never used as input to the VLM.
            media = torch.ones(
                1, 3, self.transform["image"].size, self.transform["image"].size
            )
            media_type = "text"

        tokenized_sample = self.mllm_tokenizer(
            conversations=conversations, media=media, media_type=media_type
        )
        out = (
            {
                "media": media,
                "text_ids": tokenized_sample.text_ids,
                "response_pos": tokenized_sample.response_pos,
                "image_pos": tokenized_sample.image_pos,
                "num_image_chunks": tokenized_sample.num_media_chunks,
                "media_type": media_type,
            }
            if tokenized_sample.is_valid
            else None
        )  # None will be skipped in training
        return out

    def get_conversation(
        self, caption: str, prompt: str = None
    ) -> List[Dict[str, str]]:
        """
        Converts plain caption to conversation.

        Args:
            caption (str): plain caption

        Returns:
            List[Dict[str, str]]: conversation
        """
        conversations = [
            {"from": "human", "value": prompt if prompt is not None else ""},
            {"from": "assistant", "value": caption},
        ]
        return conversations
