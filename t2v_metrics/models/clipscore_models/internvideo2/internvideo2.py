import torch
import os

import os.path as osp
from copy import deepcopy
from importlib import import_module

from ...video_utils import get_video_details, load_frames_from_video
import numpy as np

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .multi_modality.tasks.shared_utils import setup_model
from .multi_modality.models import InternVideo2_Stage2

from .multi_modality.utils.config_utils import setup_main

import datetime
import logging
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from ...video_utils import get_video_details, load_frames_from_video

from .multi_modality.models.criterions import get_sim

logger = logging.getLogger(__name__)


def extract_text_feats(texts, max_txt_l, tokenizer, model, device, return_ids=False):
    text_feats = []
    text_atts = []

    if return_ids:
        text_ids = []

    text_input = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_txt_l,
        return_tensors="pt",
    ).to(device)  # NOTE not need to cast

    text_feat = model.encode_text(text_input)[0]
    text_feats.append(text_feat)
    text_atts.append(text_input.attention_mask)
    if return_ids:
        text_ids.append(text_input.input_ids)

    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    if return_ids:
        text_ids = torch.cat(text_ids, dim=0)
        return text_feats, text_atts, text_ids
    else:
        return text_feats, text_atts


def extract_vision_feats(image_paths, transforms, model, device, num_frames=4):
    cast_dtype = None

    image_feats_all = []
    pooled_image_feats_all = []

    image = []
    for data_path in image_paths:
        total_frames, original_fps, video_duration = get_video_details(data_path)
        uniform_sampling = min(num_frames, total_frames)
        all_indices = np.linspace(0, total_frames - 1, uniform_sampling, dtype=int)
        frames = load_frames_from_video(data_path, all_indices, "decord", True)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
        frames = transforms(frames)
        image.append(frames)
        
    image = torch.stack(image, dim=0)
    image = image.to(device, dtype=cast_dtype, non_blocking=True)
    image_feat, pooled_image_feat = model.encode_vision(image, test=True)
    # if len(pooled_image_feat.shape) == 2:
    pooled_image_feat = pooled_image_feat.unsqueeze(1)  # make av_fusion happy
    # if config.evaluation.eval_frame_ensemble == "concat":
        # if len(image_feat.shape) == 4:
        #     image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
    image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
    # import pdb; pdb.set_trace()
    # else:
    #     assert config.video_input.num_frames == 1, "only support single-frame"
    #     assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
    # if config.evaluation.eval_offload:
    #     image_feats_all.append(image_feat.cpu())
    #     pooled_image_feats_all.append(pooled_image_feat.cpu())
    # else:
    image_feats_all.append(image_feat)
    pooled_image_feats_all.append(pooled_image_feat)

    image_feats_all = torch.cat(image_feats_all, dim=0)
    pooled_image_feats_all = torch.cat(pooled_image_feats_all, dim=0)

    return image_feats_all, pooled_image_feats_all


def create_transforms(image_res=224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)
    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_res, image_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )
    return test_transform


def download_internvideo2(model_name, pretrained_name, cache_dir, device="cuda"):
    repo_id = f"zhiqiulin/{model_name}"
    filename = f"{model_name}.pth"
    model_path = os.path.join(cache_dir, model_name) + ".pth"
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        os.system(f"wget -O {model_path} {hf_url}")
        # if download fails raise an error
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            raise FileNotFoundError(f"Model not found at {hf_url} because the download failed")

    # pretrained_repo_id = f"zhiqiulin/{pretrained_name}"
    # pretrained_filename = f"{pretrained_name}.pth"
    # pretrained_path = os.path.join(cache_dir, pretrained_filename)
    # if not os.path.exists(pretrained_path) or os.path.getsize(pretrained_path) < 1000:
    #     hf_url = f"https://huggingface.co/{pretrained_repo_id}/resolve/main/{pretrained_filename}"
    #     os.system(f"wget -O {pretrained_path} {hf_url}")
    #     # if download fails raise an error
    #     if not os.path.exists(pretrained_path) or os.path.getsize(pretrained_path) < 1000:
    #         raise FileNotFoundError(f"Model not found at {hf_url} because the download failed")

    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    config_file = os.path.join(current_dir, "configs.py")  # Construct the new file path
    config = setup_main(config_file=config_file, pretrained_path=model_path)

    # is_pretrain = config.mode == "pt"
    # device = torch.device(config.device)

    # train_loaders, test_name2loaders, train_media_types = setup_dataloaders(
    #     config, mode=config.mode
    # )
    (
        model,
        # model_without_ddp,
        # optimizer,
        # scheduler,
        # scaler,
        tokenizer,
        # start_epoch,
        # global_step,
    ) = setup_model(
        config,
        model_cls=InternVideo2_Stage2,
        add_decoder=False,
        pretrain=True,
        # find_unused_parameters=False,
        device=device
    )
    model = model.to(device)
    model.eval()

    return model_path, model, tokenizer, config, create_transforms(224)


@torch.no_grad()
def evaluation(
    texts,
    image_paths,
    transforms,
    model,
    tokenizer,
    device,
    num_frames=4,
    max_txt_l=40,
):
    model.eval()

    text_feats, text_atts = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device
    )  # (bsz, Lt, d), (bsz, Lt)

    image_feats, pooled_image_feats = extract_vision_feats(
        image_paths, transforms, model, device, num_frames=num_frames
    )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
    pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)
    i2t_scores, _ = get_sim(
        model.vision_proj(pooled_image_feats), model.text_proj(text_feats[:, 0])
    )

    text_encoder = model.get_text_encoder()
    encoder_output = image_feats[:, 0].to(device, non_blocking=True)
    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device, non_blocking=True)
    output = text_encoder(
        encoder_embeds=text_feats,
        attention_mask=text_atts,
        encoder_hidden_states=encoder_output,
        encoder_attention_mask=encoder_att,
        return_dict=True,
        mode="fusion",
    )

    itm_outputs = output.last_hidden_state[:, 0]
    itm_embeds = torch.cat([itm_outputs], dim=0)
    itm_scores = model.itm_head(itm_embeds)[:, 1]

    return (
        itm_scores,
        i2t_scores.diagonal()
    )
