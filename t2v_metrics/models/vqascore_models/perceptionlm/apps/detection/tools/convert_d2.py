# Modified from https://github.com/baaivision/EVA/blob/master/EVA-01/eva/interpolate_patch_14to16.py
import argparse

import torch


def interpolate_pos_embed(
    checkpoint_model, key_name="pos_embed", new_patches=196, num_extra_tokens=1
):
    if key_name in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[key_name]
        if pos_embed_checkpoint.dim() == 2:
            pos_embed_checkpoint = pos_embed_checkpoint.unsqueeze(0)
        embedding_size = pos_embed_checkpoint.shape[-1]
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(new_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
        else:
            print("Position interpolate is skipped as original size equals new size")
            return
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model[key_name] = new_pos_embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert to d2 format")
    parser.add_argument("--input", default="/path/to/input.pt", type=str)
    parser.add_argument("--output", default="/path/to/input.pt", type=str)
    parser.add_argument("--prefix", default="module.visual.", type=str)
    parser.add_argument("--output_pixel", default=224, type=int)
    parser.add_argument("--output_patch_size", default=16, type=int)
    parser.add_argument("--num_extra_tokens", default=0, type=int)
    parser.add_argument("--keep_pe", action="store_true")
    args = parser.parse_args()

    checkpoint_ori = torch.load(args.input, map_location=torch.device("cpu"))[
        "state_dict"
    ]
    checkpoint = {}

    prefix = args.prefix
    for k, v in checkpoint_ori.items():
        if k.startswith(prefix):
            checkpoint[k[len(prefix) :]] = v

    # interpolate patch_embed
    patch_embed = checkpoint["conv1.weight"]
    C_o, C_in, H, W = patch_embed.shape
    if H != args.output_patch_size or W != args.output_patch_size:
        patch_embed = torch.nn.functional.interpolate(
            patch_embed.float(),
            size=(args.output_patch_size, args.output_patch_size),
            mode="bicubic",
            align_corners=False,
        )
    checkpoint["conv1.weight"] = patch_embed

    # interpolate pos_embed too
    if not args.keep_pe:
        interpolate_pos_embed(
            checkpoint,
            key_name="positional_embedding",
            new_patches=(args.output_pixel / args.output_patch_size)
            * (args.output_pixel / args.output_patch_size),
            num_extra_tokens=args.num_extra_tokens,
        )
    else:
        positional_embedding = checkpoint["positional_embedding"].unsqueeze(0)
        checkpoint["positional_embedding"] = positional_embedding

    print("======== new state_dict ========")
    for k, v in list(checkpoint.items()):
        print(k, "        ", v.shape)

    torch.save({"model": checkpoint}, args.output)

"""
python3 tools/convert_d2.py --input /checkpoint/vision_encoder/pev1/pe_core_G14_448.pt --keep_pe --output /checkpoint/vision_encoder/pev1/pe_core_G14_448_16patch.pt
python3 tools/convert_d2.py --input /checkpoint/vision_encoder/pev1/pe_spatial_G14_448.pt --keep_pe --output /checkpoint/vision_encoder/pev1/pe_spatial_G14_16patch.pth

python3 tools/convert_d2.py --input /checkpoint/vision_encoder/pev1/pe_spatial_G14_448.pt --output_pixel 224 --output /checkpoint/vision_encoder/pev1/pe_spatial_G14_448_16patch224pix.pth
python3 tools/convert_d2.py --input /checkpoint/vision_encoder/pev1/pe_spatial_G14_448.pt --output_pixel 384 --output /checkpoint/vision_encoder/pev1/pe_spatial_G14_448_16patch384pix.pth

"""
