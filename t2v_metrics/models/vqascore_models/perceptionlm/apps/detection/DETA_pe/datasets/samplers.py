# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from codes in torch.utils.data.distributed
# ------------------------------------------------------------------------

import json
import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist

from fvcore.common.timer import Timer
from lvis import LVIS
from torch.utils.data.sampler import Sampler


def load_dataset_dicts(json_file):
    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        print("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    imgs_anns = list(zip(imgs, anns))
    print(
        "Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file)
    )
    dataset_dicts = []

    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        image_id = record["image_id"] = img_dict["id"]
        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {}
            # Convert 1-indexed to 0-indexed
            obj["category_id"] = anno["category_id"] - 1

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh, sqrt=True):
    # 1. For each category c, compute the fraction of images that contain it: f(c)
    category_freq = defaultdict(int)
    for dataset_dict in dataset_dicts:  # For each image (without repeats)
        cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
        for cat_id in cat_ids:
            category_freq[cat_id] += 1
    num_images = len(dataset_dicts)
    for k, v in category_freq.items():
        category_freq[k] = v / num_images

    # 2. For each category c, compute the category-level repeat factor:
    #    r(c) = max(1, sqrt(t / f(c)))
    category_rep = {
        cat_id: max(
            1.0,
            (
                math.sqrt(repeat_thresh / cat_freq)
                if sqrt
                else (repeat_thresh / cat_freq)
            ),
        )
        for cat_id, cat_freq in category_freq.items()
    }
    for cat_id in sorted(category_rep.keys()):
        print(
            f"Cat ID {cat_id}: freq={category_freq[cat_id]:.2f}, rep={category_rep[cat_id]:.2f}"
        )

    # 3. For each image I, compute the image-level repeat factor:
    #    r(I) = max_{c in I} r(c)
    rep_factors = []
    for dataset_dict in dataset_dicts:
        cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
        rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
        rep_factors.append(rep_factor)

    return torch.tensor(rep_factors, dtype=torch.float32)


class RepeatFactorTrainingSampler(Sampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        local_rank=None,
        local_size=None,
        shuffle=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

        json_file = (
            "/checkpoint/onevision/peizesun/public_data/d2_data/lvis/lvis_v1_train.json"
        )
        dataset_dicts = load_dataset_dicts(json_file)
        repeat_factors = repeat_factors_from_category_frequency(
            dataset_dicts, repeat_thresh=0.001
        )
        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            rfs_indices = self._get_epoch_indices(g)
            # deterministically shuffle based on epoch
            randperm = torch.randperm(len(rfs_indices), generator=g)
            indices = rfs_indices[randperm].tolist()
        else:
            g = torch.Generator()
            g.manual_seed(0)
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            rfs_indices = self._get_epoch_indices(g)
            indices = rfs_indices.tolist()

        # add extra samples to make it evenly divisible
        if self.total_size > len(indices):
            indices += indices[: (self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset : offset + self.num_samples]
            assert len(indices) == self.num_samples

            return iter(indices)
        else:
            self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
            indices += indices[: (self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset : offset + self.num_samples]
            assert len(indices) == self.num_samples

            return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        local_rank=None,
        local_size=None,
        shuffle=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class NodeDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        local_rank=None,
        local_size=None,
        shuffle=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_size is None:
            local_size = int(os.environ.get("LOCAL_SIZE", 1))
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.num_parts = local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = [i for i in indices if i % self.num_parts == self.local_rank]

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # subsample
        indices = indices[
            self.rank
            // self.num_parts : self.total_size_parts : self.num_replicas
            // self.num_parts
        ]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
