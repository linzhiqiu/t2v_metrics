# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

from core.data.data_collators import MllmPaddingCollator
from core.data.data_mixer import DatasetMixer, PersistentDataLoader
from core.data.preprocessor import VisionPreprocessor
from core.transforms.image_transform import get_image_transform
from core.transforms.region_transform import get_region_transform
from core.transforms.video_transform import get_video_transform

logger = logging.getLogger(__name__)


@dataclass
class DataloadArgs:
    datamix: str = (
        "dummy_image:1,dummy_multi_image:1,dummy_image_region:1,dummy_video:1,dummy_text:1,dummy_stc_RDCap:1,dummy_stc_RCap:1,dummy_stc_RTLoc:1"
    )
    batch_size: int = 2
    seed: int = 42
    image_res: Optional[int] = None
    max_num_tiles: Optional[int] = None
    vision_input_type: Optional[str] = None
    num_workers: Optional[int] = None
    tokenizer_path: Optional[str] = None
    tokenizer_name: Optional[str] = None
    conversation_format: Optional[str] = None
    patch_size: Optional[int] = None
    seq_len: Optional[int] = None
    max_video_frames: Optional[int] = None
    show_first_batch: Optional[bool] = False


def get_rank_position(positions, rank, workers, world_size):
    if positions is not None and rank in positions:
        if positions["num_workers"] != workers or positions["world_size"] != world_size:
            logger.warning(
                f"Checkpoint resumed with different number of total dataloader workers. Dataloaders have been reset. "
                f"num_workers: {positions['num_workers']} -> {workers}, "
                f"world_size: {positions['world_size']} -> {world_size}"
            )
            return None
        return positions[rank]
    return None


def get_dataloader(
    args,
    dp_rank,
    dp_world_size,
    dataset_configs: Dict[str, Any],
    tokenizer=None,
    positions=None,
):
    vision_input_type = args.vision_input_type
    image_res = args.image_res
    max_num_tiles = args.max_num_tiles
    max_video_frames = args.max_video_frames

    preprocessor = partial(
        VisionPreprocessor,
        transform={
            "image": get_image_transform(
                vision_input_type=vision_input_type,
                image_res=image_res,
                max_num_tiles=max_num_tiles,
            ),
            "video": get_video_transform(image_res=image_res),
            "region": get_region_transform(),
        },
        tokenizer=tokenizer,
        max_video_frames=max_video_frames,
    )

    dataset = DatasetMixer(
        args.datamix,
        global_rank=dp_rank,
        world_size=dp_world_size,
        seed=args.seed,
        preprocessors=[preprocessor],
        dataset_configs=dataset_configs,
    )

    # Create the dataloader
    dataloader = PersistentDataLoader(
        dataset,
        args.batch_size,
        args.num_workers,
        collate_fn=MllmPaddingCollator(
            tokenizer,
            show_first_batch=args.show_first_batch,
        ),
        positions=get_rank_position(
            positions, dp_rank, args.num_workers, dp_world_size
        ),
    )

    return dataloader
