from typing import List
import torch
from ..model import ScoreModel
from ...constants import HF_CACHE_DIR

from .languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor


LANGUAGEBIND_VIDEO_CLIP_MODELS = ["languagebind-video-v1.5-ft", "languagebind-video-ft", "languagebind-video-v1.5", "languagebind-video", 
                                #   "languagebind-video-v1.5-huge-ft"
                                  ]
REPO_ID = "LanguageBind"
INTERNVIDEO2_CLIP_PRETRAINED = {
    "languagebind-video-v1.5-ft": {
        "model": "LanguageBind_Video_V1.5_FT",
    },
    "languagebind-video-ft": {
        "model": "LanguageBind_Video_FT",
    },
    "languagebind-video-v1.5": {
        "model": "LanguageBind_Video_V1.5",
    },
    "languagebind-video": {
        "model": "LanguageBind_Video",
    },
    # "languagebind-video-v1.5-huge-ft": {
    #     "model": "LanguageBind_Video_V1.5_Huge_FT",
    # },
}


class LanguageBindVideoCLIPScoreModel(ScoreModel):
    "A wrapper for InternVideo2 CLIPScore models"
    video_mode = "direct"
    def __init__(
        self,
        model_name="languagebind-video-v1.5-ft",
        device="cuda",
        cache_dir=HF_CACHE_DIR,
    ):
        assert model_name in LANGUAGEBIND_VIDEO_CLIP_MODELS
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform"""
        model_type = INTERNVIDEO2_CLIP_PRETRAINED[self.model_name]["model"]
        pretrained_ckpt = f'{REPO_ID}/{model_type}'
        self.model = LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir=self.cache_dir)
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt, cache_dir=self.cache_dir)
        self.video_process = LanguageBindVideoProcessor(self.model.config, self.tokenizer)
        self.num_frames = 8 # All models have 8 frames (12 frames models are not released as of Feb 2025)
        self.model.eval()

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device"""
        # Simply verify if all are videos ending with video formats such as mp4
        if not all([img.endswith((".mp4", ".avi")) for img in image]):
            raise ValueError("All images must be video files")
        return image

    @torch.no_grad()
    def forward(
        self, images: List[str], texts: List[str], num_frames: int = 8
    ) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)"""
        assert len(images) == len(texts)
        if num_frames != self.num_frames:
            raise ValueError(
                f"num_frames must be {self.num_frames} for this model"
            )

        data = self.video_process(images, texts, return_tensors="pt")
        out = self.model(**data)
        # scores = out.text_embeds @ out.image_embeds.T
        similarity_scores = torch.sum(out.text_embeds * out.image_embeds, dim=1)
        return similarity_scores
        # import pdb; pdb.set_trace()
        # return scores.diagonal()
