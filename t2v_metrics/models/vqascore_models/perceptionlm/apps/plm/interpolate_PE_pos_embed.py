# python apps/plm/interpolate_PE_pos_embed.py \
#     --old_image_size 336 \
#     --new_image_size 448 \
#     --patch_size 14 \
#     --input_model_path facebook/PE-Core-L14-336/model.pt \
#     --output_model_path facebook/PE-Core-L14-336-interpolated-to-448/model.pt \
#     --use_cls_token

import argparse
import os

import torch
from torch.nn import functional as F


def interpolate_positional_embedding(
    old_image_size,
    new_image_size,
    patch_size,
    input_model_path,
    output_model_path,
    use_cls_token=True,
):
    _sd = torch.load(input_model_path, weights_only=True)
    if "state_dict" in _sd:
        _sd = _sd["state_dict"]
    elif "weights" in _sd:
        _sd = _sd["weights"]

    # for backwards compatibility
    _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
    if any(k.startswith("visual.") for k in _sd):
        _sd = {k.replace("visual.", ""): v for k, v in _sd.items() if "visual" in k}

    pos_embed = _sd["positional_embedding"]

    old_grid_size = old_image_size // patch_size
    new_grid_size = new_image_size // patch_size

    if use_cls_token:
        cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]
    pos_embed = (
        pos_embed.reshape(1, old_grid_size, old_grid_size, -1)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    pos_embed = F.interpolate(
        pos_embed,
        size=(new_grid_size, new_grid_size),
        mode="bilinear",
        align_corners=False,
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, 1024).contiguous()

    if use_cls_token:
        pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

    _sd["positional_embedding"] = pos_embed
    torch.save(_sd, output_model_path)
    print(f"Model saved to {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate positional embeddings for different image sizes"
    )
    parser.add_argument(
        "--old_image_size", type=int, default=336, help="Original image size"
    )
    parser.add_argument(
        "--new_image_size", type=int, default=448, help="Target image size"
    )
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size")
    parser.add_argument(
        "--input_model_path",
        type=str,
        default="facebook/PE-Core-L14-336/model.pt",
        help="Input model path",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="facebook/PE-Core-L14-336-interpolated-to-448/model.pt",
        help="Output model path",
    )
    parser.add_argument(
        "--use_cls_token",
        action="store_true",
        default=True,
        help="Whether to use class token",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_model_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    interpolate_positional_embedding(
        args.old_image_size,
        args.new_image_size,
        args.patch_size,
        args.input_model_path,
        args.output_model_path,
        args.use_cls_token,
    )
