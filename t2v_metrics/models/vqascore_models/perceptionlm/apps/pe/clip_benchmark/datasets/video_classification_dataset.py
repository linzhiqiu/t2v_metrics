import os
import random

import cv2
import decord
import torch
from PIL import Image
from torch.utils.data import Dataset


class VideoClassificationDataset(Dataset):
    def __init__(self, dataset_dir_path, task_config, preprocessor, num_frames=8):
        self.dataset_dir_path = dataset_dir_path
        self.labels_txt = task_config["labels"]
        self.media_dir_path = os.path.join(dataset_dir_path, task_config["media"])
        self.class_ids, self.classes = self.get_class_info()

        self.media_paths = []
        self.labels = []
        self.label_ids = []

        for j, (class_id, class_name) in enumerate(zip(self.class_ids, self.classes)):
            class_dir_path = os.path.join(self.media_dir_path, class_id)
            for i, video_file_name in enumerate(os.listdir(class_dir_path)):
                video_path = os.path.join(class_dir_path, video_file_name)
                self.media_paths.append(video_path)
                self.labels.append(class_name)
                self.label_ids.append(j)

        self.preprocessor = preprocessor
        self.num_frames = num_frames

    def get_class_info(self):
        class_ids = [
            dir_name
            for dir_name in os.listdir(self.media_dir_path)
            if os.path.isdir(os.path.join(self.media_dir_path, dir_name))
        ]

        if self.labels_txt:
            labels_txt_path = os.path.join(self.dataset_dir_path, self.labels_txt)
            id_to_class_name = {}
            with open(labels_txt_path, "r") as f:
                for line in f:
                    id, class_name = line.strip().split(",")
                    id_to_class_name[id] = class_name
            class_names = [id_to_class_name[id] for id in class_ids]
        else:
            class_names = class_ids

        def clean_label(label: str) -> str:
            """
            Return a label without spaces or parenthesis
            """
            for c in "()":
                label = label.replace(c, "")
            return label.strip("_")

        class_names = [clean_label(label) for label in class_names]

        return class_ids, class_names

    def __len__(self):
        return len(self.media_paths)

    def __getitem__(self, index):
        while True:
            media_path = self.media_paths[index]
            class_name = self.labels[index]
            class_id = self.label_ids[index]

            try:
                images = self._load_video(media_path)

                images = [
                    (
                        self.preprocessor(image.convert("RGB"))
                        if image.mode == "L"
                        else self.preprocessor(image)
                    )
                    for image in images
                ]
                break
            except Exception as e:
                print(f"{e}, skipping {media_path}.")
                index = random.randint(0, len(self.media_paths) - 1)

        # Returns a list of images and one class_id. The model will need to aggregate across the list of images to make a prediction.
        return images, class_id

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
