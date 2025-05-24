import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from apps.plm.transformer import LMTransformer, LMTransformerArgs
from core.args import dataclass_from_dict
from core.checkpoint import load_from_checkpoint


def build_model(
    ref_model_path: str,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
):
    ckpt_path = Path(ref_model_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    model = model_cls(model_args)
    return model


def main():
    parser = argparse.ArgumentParser(description="Consolidate PLM checkpoints")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint directory to consolidate",
    )
    args = parser.parse_args()

    model = build_model(ref_model_path=args.ckpt)
    load_from_checkpoint(
        ckpt_dir=args.ckpt,
        model=model,
        optimizer=None,
        model_key="model",
    )

    consolidated_model_state_dict = model.state_dict()
    output_file = os.path.join(args.ckpt, "consolidated.pth")

    # Save the consolidated model state_dict using torch.save
    print(f"Saving consolidated model state_dict to: {output_file}")
    torch.save(consolidated_model_state_dict, output_file)
    print("Consolidated checkpoint saved successfully.")


if __name__ == "__main__":
    main()
