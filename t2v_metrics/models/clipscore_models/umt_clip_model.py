from typing import List
import torch
from ..model import ScoreModel
from ...constants import HF_CACHE_DIR

from .umt.umt import evaluation, download_umt

UMT_CLIP_MODELS = ["umt-b16-25m-clip", "umt-l16-25m-clip"]

UMT_CLIP_PRETRAINED = {
    "umt-b16-25m-clip": {
        "model": "umt_b16_25m",
        "pretrained": "b16_ptk710_f8_res224",
    },
    "umt-l16-25m-clip": {
        "model": "umt_l16_25m",
        "pretrained": "l16_ptk710_f8_res224",
    },
}

class UMTCLIPScoreModel(ScoreModel):
    "A wrapper for UMT models"
    video_mode = "direct"
    def __init__(
        self, model_name="umt-b16-25m-clip", device="cuda", cache_dir=HF_CACHE_DIR
    ):
        assert model_name in UMT_CLIP_MODELS
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        model_name = UMT_CLIP_PRETRAINED[self.model_name]["model"]
        pretrained_ckpt_name = UMT_CLIP_PRETRAINED[self.model_name]["pretrained"]
        (
            self.model_path,
            self.pretrained_ckpt_path,
            self.model,
            self.tokenizer,
            self.config,
            self.transforms,
        ) = download_umt(model_name, pretrained_ckpt_name, self.cache_dir, device=self.device)

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        # Simply verify if all are videos ending with video formats such as mp4
        if not all([img.endswith((".mp4", ".avi")) for img in image]):
            raise ValueError("All images must be video files")
        return image

    @torch.no_grad()
    def forward(self,
                images: List[str],
                texts: List[str],
                num_frames: int=4) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        images = self.load_images(images)
        assert len(images) == len(texts)
        if num_frames != self.config.num_frames:
            raise ValueError(f"num_frames must be {self.config.num_frames} for this model")

        _, i2t_scores = evaluation(
            texts,
            images,
            self.transforms,
            self.model,
            self.tokenizer,
            self.device,
            num_frames=num_frames,
            max_txt_l=self.config.max_txt_l,
        )
        return i2t_scores
