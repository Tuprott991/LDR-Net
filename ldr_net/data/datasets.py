from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from torchvision.io import read_image
except Exception as exc:  # pragma: no cover
    read_image = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


def _to_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _load_image(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image
    elif isinstance(image, (str, Path)):
        if read_image is None:
            raise ImportError(
                "torchvision is required to load image paths in LDRNetDataset"
            ) from _TORCHVISION_IMPORT_ERROR
        tensor = read_image(str(image))
    else:
        tensor = torch.as_tensor(image)
    original_is_float = tensor.is_floating_point()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3) and tensor.shape[0] not in (1, 3):
        tensor = tensor.permute(2, 0, 1)

    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 2 or 3 dims, got {tuple(tensor.shape)}")

    if tensor.shape[0] != 1:
        tensor = tensor.float().mean(dim=0, keepdim=True)
    else:
        tensor = tensor.float()

    if not original_is_float and tensor.max().item() > 1.0:
        tensor = tensor / 255.0

    return tensor.contiguous()


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    return torch.stack((cx, cy, w, h), dim=-1)


def _normalize_boxes(boxes: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    h, w = size.tolist()
    scale = boxes.new_tensor([w, h, w, h])
    return boxes / scale.clamp(min=1.0)


def _convert_boxes(
    boxes: Any,
    image_size: torch.Tensor,
    source_format: str = "xyxy",
    target_format: str = "cxcywh",
    normalize: bool = True,
) -> torch.Tensor:
    tensor = _to_tensor(boxes, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    if tensor.ndim != 2 or tensor.shape[-1] != 4:
        raise ValueError("Boxes must have shape [N, 4].")

    source_format = source_format.lower()
    target_format = target_format.lower()
    if source_format == "xyxy" and target_format == "cxcywh":
        tensor = _xyxy_to_cxcywh(tensor)
    elif source_format != target_format:
        raise ValueError(f"Unsupported box conversion: {source_format} -> {target_format}")

    if normalize:
        tensor = _normalize_boxes(tensor, image_size)
    return tensor


def _as_mask_tensor(masks: Any) -> torch.Tensor:
    tensor = _to_tensor(masks)
    if tensor.numel() == 0:
        return tensor.reshape(0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def _deduplicate_boxes_and_labels(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    precision: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0 or labels.numel() == 0:
        return boxes, labels

    grouped: dict[tuple[float, float, float, float], list[int]] = {}
    order: list[tuple[float, float, float, float]] = []

    for box, label in zip(boxes.tolist(), labels.tolist()):
        key = tuple(round(float(value), precision) for value in box)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(int(label))

    if len(order) == len(boxes):
        return boxes, labels

    dedup_boxes = []
    dedup_labels = []
    for key in order:
        votes = grouped[key]
        label_counts: dict[int, int] = {}
        for label in votes:
            label_counts[label] = label_counts.get(label, 0) + 1
        chosen_label = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        dedup_boxes.append(list(key))
        dedup_labels.append(chosen_label)

    return (
        boxes.new_tensor(dedup_boxes, dtype=torch.float32),
        labels.new_tensor(dedup_labels, dtype=torch.long),
    )


class LDRNetDataset(Dataset):
    """Generic dataset wrapper for LDR-Net training."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]] | str | Path,
        image_key: str = "image",
        target_key: Optional[str] = None,
        image_root: Optional[str | Path] = None,
        image_size: Optional[int] = None,
        deduplicate_boxes: bool = True,
        transforms: Optional[Callable[[torch.Tensor, Dict[str, Any]], tuple[torch.Tensor, Dict[str, Any]]]] = None,
        box_format: str = "xyxy",
        normalize_boxes: bool = True,
    ) -> None:
        if isinstance(samples, (str, Path)):
            samples = self._load_samples(samples)
        self.samples = list(samples)
        self.image_key = image_key
        self.target_key = target_key
        self.image_root = Path(image_root) if image_root is not None else None
        self.image_size = image_size
        self.deduplicate_boxes = deduplicate_boxes
        self.transforms = transforms
        self.box_format = box_format
        self.normalize_boxes = normalize_boxes

    @staticmethod
    def _load_samples(path: str | Path) -> list[Mapping[str, Any]]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        if path.suffix.lower() == ".jsonl":
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("samples", "data", "items"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError("JSON dataset input must resolve to a list of sample records.")
        return payload

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "LDRNetDataset":
        return cls(path, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, value: Any) -> Any:
        if isinstance(value, (str, Path)) and self.image_root is not None:
            return self.image_root / value
        return value

    def _extract_target(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if self.target_key is None:
            return {k: v for k, v in sample.items() if k != self.image_key}
        target = sample.get(self.target_key, {})
        if not isinstance(target, Mapping):
            raise TypeError(f"Expected mapping under target key '{self.target_key}'.")
        return dict(target)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_value = sample.get(self.image_key)
        if image_value is None:
            raise KeyError(f"Sample at index {index} is missing image key '{self.image_key}'.")
        image = _load_image(self._resolve_image_path(image_value))
        if self.image_size is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        _, height, width = image.shape
        image_size = torch.tensor([height, width], dtype=torch.float32)

        target: Dict[str, Any] = self._extract_target(sample)
        has_boxes = bool(target.get("has_boxes", target.get("box_annotated", True)))
        has_disease_labels = bool(target.get("has_disease_labels", target.get("disease_annotated", True)))
        has_masks = bool(target.get("has_masks", target.get("mask_annotated", "masks" in target)))
        target["has_boxes"] = has_boxes
        target["has_disease_labels"] = has_disease_labels
        target["has_masks"] = has_masks
        target["image_size"] = image_size

        boxes = target.get("boxes", None)
        if boxes is None:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        else:
            target["boxes"] = _convert_boxes(
                boxes,
                image_size=image_size,
                source_format=target.get("box_format", self.box_format),
                target_format="cxcywh",
                normalize=bool(target.get("normalize_boxes", self.normalize_boxes)),
            )

        labels = target.get("labels", None)
        if labels is None:
            target["labels"] = torch.zeros((0,), dtype=torch.long)
        else:
            target["labels"] = _to_tensor(labels, dtype=torch.long).reshape(-1)

        if self.deduplicate_boxes:
            target["boxes"], target["labels"] = _deduplicate_boxes_and_labels(
                target["boxes"],
                target["labels"],
            )

        disease_labels = target.get("disease_labels", None)
        if disease_labels is None:
            target["disease_labels"] = None
        else:
            disease_tensor = _to_tensor(disease_labels)
            target["disease_labels"] = disease_tensor.float().reshape(-1)

        masks = target.get("masks", None)
        if masks is None:
            target["masks"] = None
        else:
            target["masks"] = _as_mask_tensor(masks)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class CXRDetectionDiseaseDataset(LDRNetDataset):
    """Chest X-ray dataset with joint lesion and disease labels."""

    def __init__(
        self,
        annotations_path: str | Path,
        image_root: str | Path = ".",
        image_size: int = 1024,
        deduplicate_boxes: bool = True,
    ):
        super().__init__(
            samples=annotations_path,
            image_key="image_path",
            target_key=None,
            image_root=image_root,
            image_size=image_size,
            deduplicate_boxes=deduplicate_boxes,
            box_format="cxcywh",
            normalize_boxes=False,
        )

    def _extract_target(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        target = dict(sample)
        target.pop("image_path", None)
        target["has_boxes"] = bool(target.get("has_detection", target.get("has_boxes", True)))
        target["has_disease_labels"] = target.get("disease_labels") is not None
        return target


class SyntheticCXRDataset(Dataset):
    """Small synthetic dataset used for CPU smoke tests."""

    def __init__(
        self,
        num_samples: int,
        image_size: int,
        num_lesions: int,
        num_diseases: int,
        max_boxes: int = 4,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_lesions = num_lesions
        self.num_diseases = num_diseases
        self.max_boxes = max_boxes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        gen = torch.Generator().manual_seed(index)
        image = torch.rand((1, self.image_size, self.image_size), generator=gen)

        num_boxes = int(torch.randint(low=0, high=self.max_boxes + 1, size=(1,), generator=gen).item())
        if num_boxes > 0:
            cxcy = torch.rand((num_boxes, 2), generator=gen) * 0.8 + 0.1
            wh = torch.rand((num_boxes, 2), generator=gen) * 0.3 + 0.05
            boxes = torch.cat([cxcy, wh.clamp(max=0.9)], dim=-1)
            labels = torch.randint(0, self.num_lesions, (num_boxes,), generator=gen, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        disease_labels = torch.zeros(self.num_diseases, dtype=torch.float32)
        active = torch.randint(0, self.num_diseases, (max(1, min(3, self.num_diseases // 4)),), generator=gen)
        disease_labels[active.unique()] = 1.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "disease_labels": disease_labels,
            "has_boxes": bool(num_boxes > 0),
            "has_disease_labels": True,
            "sample_id": f"synthetic_{index:05d}",
            "image_size": torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
        }
        return image, target
