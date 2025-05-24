import io
import logging
import pickle
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm

# from open_clip import image_to_device


def evaluate(model, dataloader, device, visargs, amp=True, args=None):
    autocast = torch.cuda.amp.autocast if amp else suppress
    model.visual.output_tokens = True

    if visargs.delete_post_ln:
        model.visual.ln_post = nn.Identity()
    if visargs.extract_layer is not None and visargs.extract_layer != -1:
        model.visual.transformer.resblocks = model.visual.transformer.resblocks[
            : (visargs.extract_layer + 1)
        ]

    if visargs.attn_rollout is not None:
        for module in model.visual.transformer.resblocks:
            if hasattr(module, "attn"):
                module.attn.forward = module.attn.forward_with_attn

    with torch.no_grad():
        for images, target in dataloader:
            old_images = images
            # images = image_to_device(images, device, torch.float32, mean=args.image_mean, std=args.image_std)
            iamges = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                if hasattr(model.visual, "trunk"):
                    x = model.visual.trunk.forward_features(images)
                else:
                    x = model.visual(images).latent

                # apply attn pooler if we want
                if visargs.attn_pooling is not None:
                    idx = {"Q": 0, "K": 1, "V": 2}[visargs.attn_pooling]
                    attn_layer = model.visual.attn_pool.attn
                    mat = attn_layer.in_proj_weight[
                        attn_layer.embed_dim * idx : attn_layer.embed_dim * (idx + 1)
                    ]
                    x = x @ mat.T

                    if attn_layer.in_proj_bias is not None:
                        x = (
                            x
                            + attn_layer.in_proj_bias[
                                None,
                                None,
                                idx
                                * attn_layer.embed_dim : (idx + 1)
                                * attn_layer.embed_dim,
                            ]
                        )

                x = x.reshape(target.shape[0], -1, x.shape[-1])

                if isinstance(images, list):
                    old_images = old_images[0]

                if (
                    hasattr(model.visual, "embed_cls_token")
                    and model.visual.embed_cls_token
                ) or (
                    hasattr(model.visual, "trunk")
                    and hasattr(model.visual.trunk, "cls_token")
                    and model.visual.trunk.cls_token
                ):
                    if isinstance(images, list):
                        x = x[:, :-1]
                        old_images = old_images.reshape(
                            target.shape[0], -1, *old_images.shape[1:]
                        )
                        old_images = old_images[:, :-1]
                    else:
                        x = x[:, 1:]

                sidelen = int(x.shape[1] ** 0.5)

                if isinstance(images, list):
                    old_images = old_images.reshape(
                        target.shape[0], sidelen, sidelen, 3, *old_images.shape[-2:]
                    )
                    psz = old_images.shape[-1]
                    old_images = old_images.permute(0, 3, 1, 4, 2, 5).reshape(
                        target.shape[0], 3, sidelen * psz, sidelen * psz
                    )

                x = x.view(x.shape[0], sidelen, sidelen, -1)
                x = x.permute(0, 3, 1, 2).contiguous()
                # convert on umpy
                x = x.cpu().numpy().astype(np.float16)

                old_images = old_images.cpu().numpy().astype(np.float16)
                old_images *= np.array(args.image_std)[None, :, None, None]
                old_images += np.array(args.image_mean)[None, :, None, None]
                data = {"image": old_images, "features": x}

                if visargs.attn_pooling is not None:
                    data["attn"] = {
                        "qkv": attn_layer.in_proj_weight.data,
                        "bias": (
                            attn_layer.in_proj_bias.data
                            if attn_layer.in_proj_bias is not None
                            else None
                        ),
                        "probe": model.visual.attn_pool.probe,
                    }

                if visargs.attn_rollout:
                    attns = [
                        module.attn.attn.mean(dim=1)
                        for module in model.visual.transformer.resblocks
                        if hasattr(module, "attn")
                    ]

                    def roll(i, rollout=None):
                        for attn in attns[i:]:
                            rollout = (
                                attn.float() @ rollout
                                if rollout is not None
                                else attn.float()
                            )
                        return rollout

                    # rollout = torch.stack([roll(i) for i in range(len(attns))], dim=1)
                    rollout = torch.stack(attns, dim=1)
                    data["rollout"] = rollout.cpu().numpy().astype(np.float16)

                bytes_str = pickle.dumps(data)

                print(f"<<<<<<<<<<<<<")
                print(bytes_str)
                print(f">>>>>>>>>>>>>")

            break

    exit()
    return None
