"""Feature pyramid network used by the lesion-aware model."""

from __future__ import annotations

from collections import OrderedDict
from typing import List

from torch import Tensor, nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """Minimal FPN implementation for a fixed set of backbone stages."""

    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        )

    def forward(self, features: "OrderedDict[str, Tensor]") -> "OrderedDict[str, Tensor]":
        names = list(features.keys())
        xs = list(features.values())

        pyramid = [None] * len(xs)
        last = None
        for idx in reversed(range(len(xs))):
            lateral = self.lateral_convs[idx](xs[idx])
            if last is not None:
                lateral = lateral + F.interpolate(last, size=lateral.shape[-2:], mode="nearest")
            last = lateral
            pyramid[idx] = self.output_convs[idx](lateral)

        return OrderedDict((f"p{idx + 2}", feat) for idx, feat in enumerate(pyramid))

