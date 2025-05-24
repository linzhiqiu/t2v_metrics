# Copyright (c) Meta Platforms, Inc. and affiliates.
import math

import torch
import torch.nn.functional as F
from torch import nn

from core.utils import get_init_fn
from core.vision_projector.base import BaseProjector


class AdaptiveAvgPooling(nn.Module):
    def __init__(self, pooling_ratio=2):
        super(AdaptiveAvgPooling, self).__init__()
        self.pooling_ratio = pooling_ratio

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens

        shape = (h // self.pooling_ratio, h // self.pooling_ratio)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.adaptive_avg_pool2d(x, shape)
        x = x.flatten(2).transpose(1, 2)

        return x


class MLPProjector(BaseProjector):
    def __init__(self, args):
        super().__init__()
        self.setup_projector(args)
        self.pooling_ratio = args.pooling_ratio
        self.adaptive_avg_pool = AdaptiveAvgPooling(pooling_ratio=args.pooling_ratio)
        self.remove_vision_class_token = args.remove_vision_class_token

    def init_tensors(self):
        self.init_method(self.projector[0].weight)
        self.init_method(self.projector[0].bias)
        self.init_method(self.projector[2].weight)
        self.init_method(self.projector[2].bias)

    def setup_projector(self, args):
        self.init_method = get_init_fn(args.mlp_init, args.dim, init_depth=None)
        input_size = args.vision_model["width"]
        output_size = args.dim
        self.projector = nn.Sequential(
            nn.Linear(
                in_features=input_size,
                out_features=output_size,
                bias=True,
                dtype=torch.get_default_dtype(),
            ),
            nn.GELU(),
            nn.Linear(
                in_features=output_size,
                out_features=output_size,
                bias=True,
                dtype=torch.get_default_dtype(),
            ),
        )
