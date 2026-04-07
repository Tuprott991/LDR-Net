"""Backbone building blocks for LDR-Net.

This module intentionally keeps the feature extractor lightweight and fully
local so the project can run on CPU without any external checkpoints.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn
from torchvision import models as tvm


class ResNetBackbone(nn.Module):
    """A small ResNet feature pyramid stem.

    The network uses a torchvision ResNet-18 backbone with the first
    convolution adapted for grayscale chest radiographs by default.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        try:
            backbone = tvm.resnet18(weights=None)
        except TypeError:  # pragma: no cover - older torchvision fallback
            backbone = tvm.resnet18(pretrained=False)

        if in_channels != backbone.conv1.in_channels:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=False,
            )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @property
    def out_channels(self) -> List[int]:
        return [64, 128, 256, 512]

    def forward(self, x: Tensor) -> "OrderedDict[str, Tensor]":
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])

