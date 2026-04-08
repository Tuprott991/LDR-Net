"""Backbone building blocks for LDR-Net."""

from __future__ import annotations

from collections import OrderedDict
from typing import List
import warnings

from torch import Tensor, nn
from torchvision import models as tvm
import torch


class ResNetBackbone(nn.Module):
    """A small ResNet feature pyramid stem."""

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


class HFConvNeXtV2Backbone(nn.Module):
    """ConvNeXtV2 backbone loaded from a Hugging Face checkpoint."""

    DEFAULT_MODEL_NAME = "shreydan/CheXpert-5-convnextv2-tiny-384"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        in_channels: int = 1,
        local_files_only: bool = False,
        fallback_to_resnet: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.in_channels = in_channels
        self.local_files_only = local_files_only
        self.fallback = None

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        try:
            from transformers import AutoModelForImageClassification

            classifier_model = AutoModelForImageClassification.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            if hasattr(classifier_model, "convnextv2"):
                self.backbone = classifier_model.convnextv2
            elif hasattr(classifier_model, "base_model"):
                self.backbone = classifier_model.base_model
            else:  # pragma: no cover - defensive
                raise AttributeError("Unsupported Hugging Face model structure for ConvNeXtV2 backbone extraction.")
            self._out_channels = [96, 192, 384, 768]
            self.channel_adapter = None
            if in_channels not in (1, 3):
                self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        except Exception as exc:
            if not fallback_to_resnet:
                raise
            warnings.warn(
                f"Falling back to torchvision ResNet backbone because '{model_name}' could not be loaded: {exc}",
                stacklevel=2,
            )
            self.fallback = ResNetBackbone(in_channels=in_channels)
            self.backbone = None
            self._out_channels = self.fallback.out_channels
            self.channel_adapter = None

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def _prepare_inputs(self, x: Tensor) -> Tensor:
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        elif self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, x: Tensor) -> "OrderedDict[str, Tensor]":
        if self.fallback is not None:
            return self.fallback(x)

        x = self._prepare_inputs(x)
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        return OrderedDict(
            [
                ("c2", hidden_states[0]),
                ("c3", hidden_states[1]),
                ("c4", hidden_states[2]),
                ("c5", hidden_states[3]),
            ]
        )


def build_backbone(
    backbone_type: str = "hf_convnextv2",
    backbone_name: str = HFConvNeXtV2Backbone.DEFAULT_MODEL_NAME,
    in_channels: int = 1,
    local_files_only: bool = False,
    fallback_to_resnet: bool = True,
) -> nn.Module:
    backbone_type = backbone_type.lower()
    if backbone_type in {"hf_convnextv2", "convnextv2", "huggingface"}:
        return HFConvNeXtV2Backbone(
            model_name=backbone_name,
            in_channels=in_channels,
            local_files_only=local_files_only,
            fallback_to_resnet=fallback_to_resnet,
        )
    if backbone_type in {"resnet", "resnet18", "torchvision"}:
        return ResNetBackbone(in_channels=in_channels)
    raise ValueError(f"Unsupported backbone_type: {backbone_type}")
