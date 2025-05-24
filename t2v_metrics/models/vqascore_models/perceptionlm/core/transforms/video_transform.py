# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from logging import getLogger
from typing import Callable, List, Tuple

import torch
from PIL import Image
from torchcodec.decoders import VideoDecoder
from torchvision.utils import draw_bounding_boxes

from core.transforms.image_transform import ImageTransform

logger = getLogger()


def get_video_transform(
    image_res: int = 224,
    normalize_img: bool = True,
) -> Tuple[Callable, int]:

    transforms = VideoTransform(
        size=image_res,
        normalize_img=normalize_img,
    )

    return transforms


class VideoTransform(ImageTransform):
    def __init__(
        self,
        size: int = 224,
        normalize_img: bool = True,
    ) -> None:
        super().__init__(
            size=size,
            normalize_img=normalize_img,
        )

    def __call__(self, video_info: tuple, sampling_fps: int = 1):
        video_path, max_frames, s, e, bbox_map = video_info

        frames, sample_pos = self.load_video(
            video_path,
            max_frames=max_frames,
            sampling_fps=sampling_fps,
            s=s,
            e=e,
        )

        if bbox_map:
            bbox_dict_map = {}
            for idx_pos, pos in enumerate(sample_pos):
                if str(pos) in bbox_map and bbox_map[str(pos)] is not None:
                    bbox_dict_map[idx_pos] = bbox_map[str(pos)]
            if len(bbox_dict_map) > 0:
                frames = self.draw_bounding_boxes(frames, bbox_dict_map)

        return super()._transform_torch_tensor(frames)

    def _process_multiple_images(self, image_paths: List[str]):
        images = [Image.open(path).convert("RGB") for path in image_paths]
        processed_images = []
        for image in images:
            image, (w, h) = super().__call__(image)
            processed_images.append(image)
        processed_images = torch.cat(processed_images, dim=0)
        return processed_images, (w, h)

    def _process_multiple_images_pil(self, images: List[Image.Image]):
        processed_images = []
        for image in images:
            image, (w, h) = super().__call__(image)
            processed_images.append(image)
        processed_images = torch.cat(processed_images, dim=0)
        return processed_images, (w, h)

    def load_video(self, video_path, max_frames=16, sampling_fps=1, s=None, e=None):
        """
        Loads a video from a given path and extracts frames based on specified parameters using OpenCV.

        Args:
            video_path (str): The path to the video file.
            max_frames (int, optional): The maximum number of frames to extract. Defaults to 16.
            sampling_fps (int, optional): The sampling frame rate. Defaults to 1.
            s (float, optional): The start time of the video in seconds. Defaults to None.
            e (float, optional): The end time of the video in seconds. Defaults to None.

        Returns:
            list: A list of frames extracted from the video.
        """

        if not os.path.exists(video_path):
            return

        decoder = VideoDecoder(video_path, device="cpu")
        decoder_metadata = decoder.metadata
        fps = decoder_metadata.average_fps
        total_frames = decoder_metadata.num_frames

        start_frame = 0 if s is None else int(s * fps)
        end_frame = total_frames - 1 if e is None else int(e * fps)
        end_frame = min(end_frame, total_frames - 1)

        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        elif start_frame == end_frame:
            end_frame = start_frame + 1

        sample_fps = int(sampling_fps)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(start_frame, end_frame + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_idxs = self.uniform_sample(len(all_pos), max_frames)
            sample_pos = [all_pos[i] for i in sample_idxs]
        elif len(all_pos) < max_frames:
            total_clip_frames = end_frame - start_frame + 1
            if total_clip_frames < max_frames:
                max_frames = total_clip_frames
            sample_idxs = self.uniform_sample(total_clip_frames, max_frames)
            sample_pos = [start_frame + idx for idx in sample_idxs]
        else:
            sample_pos = all_pos

        all_frames = decoder.get_frames_at(indices=sample_pos)
        all_frames = all_frames.data

        return all_frames, sample_pos

    def uniform_sample(self, m, n):
        assert n <= m
        stride = (m - 1) / (n - 1) if n > 1 else 0  # Calculate the stride
        return [int(round(i * stride)) for i in range(n)]

    def draw_bounding_boxes(self, frames, all_bboxes):
        # Assuming frames is a torch.Tensor with shape (N, C, H, W)
        N, _, _, _ = frames.shape
        frames_with_bbox = (
            frames.clone()
        )  # Clone the tensor to avoid modifying the original
        for i in range(N):
            if i in all_bboxes:
                bbox = all_bboxes[i]
                # Convert bbox to a tensor and add a batch dimension
                bbox_tensor = torch.tensor([bbox], dtype=torch.float32)
                # Draw the bounding box on the frame
                frames_with_bbox[i] = draw_bounding_boxes(
                    frames_with_bbox[i],
                    boxes=bbox_tensor,
                    colors=(255, 0, 0),  # Red color for the bounding box
                    width=4,
                )
        return frames_with_bbox
