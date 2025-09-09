import os
import random

import cv2
import decord
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class VideoRetrievalDataset(Dataset):
    def __init__(
        self,
        csv_path,
        dataset_dir,
        preprocessor,
        video_ext="mp4",
        num_frames=8,
        multi_sent=False,
    ):
        self.data = pd.read_csv(csv_path)
        self.dataset_dir = dataset_dir
        self.video_ext = video_ext

        self.preprocessor = preprocessor
        self.num_frames = num_frames
        self.multi_sent = multi_sent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id = self.data["video_id"].values[index]
        sentences = self.data["sentence"].values[index]
        if self.multi_sent:
            sentences = sentences.split("@")
        else:
            sentences = [sentences]
        video_path = os.path.join(
            self.dataset_dir, "{}.{}".format(video_id, self.video_ext)
        )

        images = self._load_video(video_path)

        images = [
            (
                self.preprocessor(image.convert("RGB"))
                if image.mode == "L"
                else self.preprocessor(image)
            )
            for image in images
        ]

        return images, sentences

    def _load_video(self, media_path):
        vr = decord.VideoReader(media_path)
        total_frames = len(vr)
        if self.num_frames == 1:
            frame_indices = [total_frames // 2]
        else:
            frame_indices = [
                int(i * (total_frames - 1) / (self.num_frames - 1))
                for i in range(self.num_frames)
            ]
        try:
            images = vr.get_batch(frame_indices).asnumpy()
        except Exception as e:
            cap = cv2.VideoCapture(media_path)
            images = []
            for pos in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    # Convert the frame from BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images.append(rgb_frame)
                else:
                    break

        images = [Image.fromarray(image) for image in images]

        return images
