# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import ABC, abstractmethod

from torch import nn


class BaseProjector(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.adaptive_avg_pool = None

    @abstractmethod
    def setup_projector(self):
        """
        Setup the vision_projector attribute in subclasses.
        """
        pass

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.projector(x)
        x = x.permute(1, 0, 2)
        if self.adaptive_avg_pool is not None:
            x = self.adaptive_avg_pool(x)
        return x
