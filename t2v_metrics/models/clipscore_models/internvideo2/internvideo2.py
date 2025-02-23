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
    config,
    num_frames=4,
    max_txt_l=40,
):
    model.eval()

    # use_dsl_for_match = config.evaluation.get('use_dsl_for_match', False)

    # dtype = torch.float
    # media_type = data_loader.dataset.media_type
    # use_subtitle = hasattr(data_loader.dataset, "use_subtitle") and data_loader.dataset.use_subtitle
    # if use_subtitle:
    #     assert media_type in ["video", "audio_video"], f"Not support media_type: {media_type}."
    #     assert hasattr(data_loader.dataset, "subtitle") and data_loader.dataset.subtitle is not None, "You don't have subtitle to use."

    # logger.info(f"Start evaluation for media_type={media_type}")
    # assert media_type in ['audio', 'video', 'audio_video'], f"Not implement evaluation of {media_type}"

    # logger.info("Computing dual encoder features...")
    # start_time = time.time()

    # this computes all features in each GPU
    # texts = data_loader.dataset.text
    # max_txt_l of eval depends on data_cofig
    # max_txt_l = data_loader.dataset.max_txt_l

    text_feats, text_atts = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device
    )  # (bsz, Lt, d), (bsz, Lt)

    # if use_subtitle:
    #     subtitle_feats, _ = extract_text_feats(
    #         data_loader.dataset.subtitle, max_txt_l, tokenizer, model, device
    #     ) # (bsz, Lt, d), (bsz, Lt)
    #     subtitle_proj = model.text_proj(subtitle_feats[:, 0]).unsqueeze(1)
    #     subtitle_feats = subtitle_feats.unsqueeze(1)

    image_feats, pooled_image_feats = extract_vision_feats(
        image_paths, transforms, model, device, num_frames=num_frames
    )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
    # logger.info("Finished vision feature extraction")
    # logger.info("Computing ITC scores [dot-product]")
    # if config.evaluation.eval_offload:
    # image_feats = image_feats.to(device, non_blocking=True) image_feats will cause OOM!!!
    pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)

    # if use_subtitle:
    #     # print(subtitle_proj.shape, pooled_image_feats.shape)
    #     i2t_scores, t2i_scores = get_sim(
    #         model.vs_fusion(torch.concat([subtitle_proj, model.vision_proj(pooled_image_feats)], dim=-1)), model.text_proj(text_feats[:, 0])
    #     )
    # else:
    i2t_scores, _ = get_sim(
        model.vision_proj(pooled_image_feats), model.text_proj(text_feats[:, 0])
    )

    # if use_dsl_for_match:
    #     logger.info("use_dsl_for_match!!!")
    #     old_i2t_scores, old_t2i_scores = i2t_scores, t2i_scores
    #     i2t_scores = old_i2t_scores * old_i2t_scores.softmax(dim=0)
    #     t2i_scores = old_i2t_scores.T * old_i2t_scores.T.softmax(dim=0)

    # num_medias = len(image_paths)

    # pooled_media_feats = pooled_image_feats
    # if use_subtitle:
    #     media_feats = torch.concat([subtitle_feats, image_feats], dim=-2)
    #     if hasattr(model, "vstm_head"):
    #         match_head = model.vstm_head
    #     else:
    #         match_head = None
    # else:
    # media_feats = image_feats
    # if hasattr(model, "itm_head"):
    match_head = model.itm_head
    # else:
    #     match_head = None

    # logger.info("Computing ITC scores [dot-product], done!")
    # i2t_scores_x = torch.full((num_medias, len(texts)), -100.0).to(
    #     device, torch.float, non_blocking=True
    # )

    # computes only part of the scores at each GPU, gather at the end
    # logger.info("Rerank dual-encoder results with cross-encoder...")
    # num_tasks = 1
    # rank = 0
    # only uses the part associated with the raw eval set
    # compute media2text #
    # step = num_medias + 1
    # start = 0
    # end = min(num_medias, start + step)

    text_encoder = model.get_text_encoder()
    # iterator = metric_logger.log_every(i2t_scores[start:end], 100, header)
    # logger.info(f"i2t_scores.shape {i2t_scores[start:end].shape}")

    # generate score for each clip, and aggregate all clip scores for a video
    # n_clip_per_video = (
    #     media_feats.shape[1] if not config.deep_fusion else media_feats[0].shape[1]
    # )

    # assert not config.deep_fusion and n_clip_per_video == 1, f"Not implemented for config.deep_fusion={config.deep_fusion} n_clip_per_video={n_clip_per_video}"

    # logger.info(
    #     f"n_clip_per_video={n_clip_per_video}, with eval_frame_ensemble={config.evaluation.eval_frame_ensemble}"
    # )
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
    itm_scores = match_head(itm_embeds)[:, 1]

    # for i, sims in enumerate(i2t_scores[start:end]):
    #     k = min(len(sims), config.evaluation.k_test)
    #     topk_sim, topk_idx = sims.topk(k=k, dim=0)

    #     clip_scores = []
    #     clip_idx = 0
    #     encoder_output = (
    #         media_feats[start + i, clip_idx].to(device, non_blocking=True)
    #         if config.evaluation.eval_offload
    #         else media_feats[start + i, clip_idx]
    #     )  # (#frm*Li, d)

    #     # new
    #     bs = 32
    #     # bs = config.batch_size_test.video
    #     itm_embeds = []
    #     if len(topk_idx) % bs != 0:
    #         left = len(topk_idx) % bs
    #         left_encoder_output = encoder_output.repeat(left, 1, 1)  # (k=128, #frm*Li, d)
    #         left_encoder_att = torch.ones(left_encoder_output.size()[:-1], dtype=torch.long).to(
    #             device, non_blocking=True
    #         )
    #     encoder_output = encoder_output.repeat(bs, 1, 1)  # (k=128, #frm*Li, d)
    #     encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
    #         device, non_blocking=True
    #     )

    #     for j in range(0, len(topk_idx), bs):
    #         if j + bs > len(topk_idx):
    #             output = text_encoder(
    #                 encoder_embeds=text_feats[topk_idx[j:]],
    #                 attention_mask=text_atts[topk_idx[j:]],
    #                 encoder_hidden_states=left_encoder_output,
    #                 encoder_attention_mask=left_encoder_att,
    #                 return_dict=True,
    #                 mode="fusion",
    #             )
    #         else:
    #             output = text_encoder(
    #                 encoder_embeds=text_feats[topk_idx[j : j + bs]],
    #                 attention_mask=text_atts[topk_idx[j : j + bs]],
    #                 encoder_hidden_states=encoder_output,
    #                 encoder_attention_mask=encoder_att,
    #                 return_dict=True,
    #                 mode="fusion",
    #             )
    #         batch_itm_embeds = output.last_hidden_state[:, 0]
    #         itm_embeds.append(batch_itm_embeds)
    #         import pdb; pdb.set_trace()
    #     itm_embeds = torch.cat(itm_embeds, dim=0)
    #     # end new

    #     score = match_head(itm_embeds)[:, 1]
    #     clip_scores.append(score)

    #     if len(clip_scores) == 1:
    #         score = clip_scores[0]
    #     else:
    #         raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

    #     i2t_scores_x[start + i, topk_idx] = score.to(i2t_scores_x.dtype)

    # for i, sims in enumerate(t2i_scores[start:end]):
    #     k = min(len(sims), config.evaluation.k_test)
    #     topk_sim, topk_idx = sims.topk(k=k, dim=0)

    #     clip_scores = []
    #     for clip_idx in range(n_clip_per_video):
    #         # new
    #         bs = 32
    #         # bs = config.batch_size_test.video
    #         itm_embeds = []
    #         for j in range(0, len(topk_idx), bs):

    #             if config.deep_fusion:
    #                 encoder_output = [
    #                     feat[topk_idx[j : j + bs].cpu(), clip_idx].to(device, non_blocking=True)
    #                     if config.evaluation.eval_offload
    #                     else feat[topk_idx[j : j + bs], clip_idx]
    #                     for feat in media_feats
    #                 ]
    #                 encoder_att = [
    #                     torch.ones(feat.size()[:-1], dtype=torch.long).to(
    #                         device, non_blocking=True
    #                     )
    #                     for feat in encoder_output
    #                 ]
    #             else:
    #                 encoder_output = (
    #                     media_feats[topk_idx[j : j + bs].cpu(), clip_idx].to(
    #                         device, non_blocking=True
    #                     )
    #                     if config.evaluation.eval_offload
    #                     else media_feats[topk_idx[j : j + bs], clip_idx]
    #                 )
    #                 encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
    #                     device, non_blocking=True
    #                 )

    #             repeat_n = (
    #                 encoder_output.shape[0]
    #                 if not config.deep_fusion
    #                 else encoder_output[0].shape[0]
    #             )
    #             output = text_encoder(
    #                 encoder_embeds=text_feats[start + i].repeat(repeat_n, 1, 1),
    #                 attention_mask=text_atts[start + i].repeat(repeat_n, 1),
    #                 encoder_hidden_states=encoder_output,
    #                 encoder_attention_mask=encoder_att,
    #                 return_dict=True,
    #                 mode="fusion",
    #             )

    #             batch_itm_embeds = output.last_hidden_state[:, 0]
    #             itm_embeds.append(batch_itm_embeds)

    #         itm_embeds = torch.cat(itm_embeds, dim=0)
    #         # end new

    #         score = match_head(itm_embeds)[:, 1]
    #         clip_scores.append(score)

    #     if len(clip_scores) == 1:
    #         score = clip_scores[0]
    #     else:
    #         raise NotImplementedError(f"len(clip_scores) == {len(clip_scores)}")

    #     t2i_scores_x[start + i, topk_idx] = score.to(t2i_scores_x.dtype)

    # logger.info("Compute over!!!")
    # if config.distributed:
    #     logger.info("Gather across GPUs!!!")
    #     # gather across GPUs
    #     dist.barrier()
    #     logger.info("dist.barrier()!!!")
    #     dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM)
    #     logger.info("dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM) over!!!")
    #     dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM)
    #     logger.info("dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM) over!!!")

    # if use_dsl_for_match:
    #     i2t_scores_dsl = i2t_scores
    #     i2t_scores_dsl_T = t2i_scores
    #     i2t_scores = old_i2t_scores
    #     t2i_scores = old_t2i_scores
    # else:
    # i2t_scores_dsl = i2t_scores.float() * i2t_scores.float().softmax(dim=0)
    # i2t_scores_dsl_T = i2t_scores.float().T * i2t_scores.float().T.softmax(dim=0)
    # else:
    #     i2t_scores_dsl = i2t_scores.float() * i2t_scores.float().softmax(dim=0)
    #     i2t_scores_dsl_T = i2t_scores.float().T * i2t_scores.float().T.softmax(dim=0)

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # logger.info(f"Evaluation time {total_time_str}")

    # if match_head is not None:
    import pdb; pdb.set_trace()
    return (
        itm_scores,
        i2t_scores.diagonal()
        # i2t_scores_x.softmax(dim=1).cpu().float().numpy(),
        # t2i_scores_x.softmax(dim=1).cpu().float().numpy(),
        # i2t_scores.softmax(dim=1).cpu().float().numpy(),
        # i2t_scores.T.softmax(dim=1).cpu().float().numpy(),
        # i2t_scores_dsl.softmax(dim=1).cpu().float().numpy(),
        # i2t_scores_dsl_T.softmax(dim=1).cpu().float().numpy()
    )
    # else:
    #     return (
    #         None,
    #         None,
    #         i2t_scores.softmax(dim=1).cpu().float().numpy(),
    #         i2t_scores.T.softmax(dim=1).cpu().float().numpy(),
    #         i2t_scores_dsl.softmax(dim=1).cpu().float().numpy(),
    #         i2t_scores_dsl_T.softmax(dim=1).cpu().float().numpy()
    #     )
