from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .matcher import HungarianMatcher, _is_box_supervised


def _is_disease_supervised(target: Mapping) -> bool:
    return bool(target.get("has_disease_labels", target.get("disease_annotated", target.get("disease_labels") is not None)))


def _is_mask_supervised(target: Mapping) -> bool:
    return bool(target.get("has_masks", target.get("mask_annotated", target.get("masks") is not None)))


def _stack_if_any(tensors):
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def _first_tensor(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    for value in outputs.values():
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
    return None


def _zero_from_outputs(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    tensor = _first_tensor(outputs)
    if tensor is None:
        return torch.tensor(0.0)
    return tensor.sum() * 0.0


def _get_src_permutation_idx(indices: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    batch_parts = []
    src_parts = []
    device = None
    for i, (src, _) in enumerate(indices):
        if src.numel() == 0:
            continue
        device = src.device
        batch_parts.append(torch.full_like(src, i))
        src_parts.append(src)
    if not batch_parts:
        if device is None:
            device = torch.device("cpu")
        return torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(batch_parts, dim=0), torch.cat(src_parts, dim=0)


class SetCriterion(nn.Module):
    """Detection and disease criterion with mixed-supervision support."""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Mapping[str, float] | None = None,
        eos_coef: float = 0.1,
        num_disease_classes: int | None = None,
        losses: Sequence[str] = ("labels", "boxes", "diseases", "masks"),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.num_disease_classes = num_disease_classes
        self.losses = tuple(losses)
        default_weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_disease": 1.0,
            "loss_mask": 1.0,
            "loss_dice": 1.0,
        }
        self.weight_dict = default_weight_dict if weight_dict is None else dict(weight_dict)

        if getattr(self.matcher, "num_classes", None) is None:
            self.matcher.num_classes = num_classes

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _stack_disease_targets(self, targets: Sequence[Mapping], device: torch.device, channels: int | None):
        tensors = []
        for target in targets:
            if not _is_disease_supervised(target):
                continue
            disease = target.get("disease_labels")
            if disease is None:
                continue
            disease_tensor = torch.as_tensor(disease, dtype=torch.float32, device=device).reshape(-1)
            if disease_tensor.numel() == 1 and channels is not None and channels > 1:
                one_hot = torch.zeros(channels, device=device)
                one_hot[int(disease_tensor.item())] = 1.0
                disease_tensor = one_hot
            elif channels is not None and disease_tensor.numel() != channels:
                raise ValueError(
                    f"Disease target has {disease_tensor.numel()} values but model expects {channels}."
                )
            tensors.append(disease_tensor)
        if not tensors:
            return None
        return torch.stack(tensors, dim=0)

    def _sigmoid_dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1).float()
        numerator = 2 * (inputs * targets).sum(-1) + 1
        denominator = inputs.sum(-1) + targets.sum(-1) + 1
        return 1 - numerator / denominator

    def loss_labels(self, outputs: Mapping[str, torch.Tensor], targets: Sequence[Mapping], indices):
        src_logits = outputs["pred_logits"]
        batch_idx, src_idx = _get_src_permutation_idx(indices)
        if src_idx.numel() == 0:
            return {"loss_ce": _zero_from_outputs(outputs)}

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        for batch_pos, (src, tgt) in enumerate(indices):
            if src.numel() == 0:
                continue
            target_classes[batch_pos, src] = targets[batch_pos]["labels"][tgt].to(device=src_logits.device, dtype=torch.int64)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs: Mapping[str, torch.Tensor], targets: Sequence[Mapping], indices):
        src_boxes = outputs["pred_boxes"]
        batch_idx, src_idx = _get_src_permutation_idx(indices)
        if src_idx.numel() == 0:
            zero = _zero_from_outputs(outputs)
            return {"loss_bbox": zero, "loss_giou": zero}

        src = src_boxes[batch_idx, src_idx]
        tgt = torch.cat(
            [targets[i]["boxes"][tgt_idx] for i, (_, tgt_idx) in enumerate(indices) if tgt_idx.numel() > 0],
            dim=0,
        ).to(device=src_boxes.device, dtype=src_boxes.dtype)

        loss_bbox = F.l1_loss(src, tgt, reduction="none").sum() / max(tgt.shape[0], 1)
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt))
        ).mean()
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_masks(self, outputs: Mapping[str, torch.Tensor], targets: Sequence[Mapping], indices):
        pred_masks = outputs.get("pred_masks")
        if pred_masks is None:
            zero = _zero_from_outputs(outputs)
            return {"loss_mask": zero, "loss_dice": zero}

        src_mask_parts = []
        tgt_mask_parts = []
        for batch_pos, (src, tgt) in enumerate(indices):
            if src.numel() == 0:
                continue
            target = targets[batch_pos]
            if not _is_mask_supervised(target) or target.get("masks") is None:
                continue
            src_mask_parts.append(pred_masks[batch_pos, src])
            tgt_mask_parts.append(target["masks"][tgt].to(device=pred_masks.device, dtype=pred_masks.dtype))

        src_masks = _stack_if_any(src_mask_parts)
        tgt_masks = _stack_if_any(tgt_mask_parts)
        if src_masks is None or tgt_masks is None:
            zero = _zero_from_outputs(outputs)
            return {"loss_mask": zero, "loss_dice": zero}

        if src_masks.ndim == 4 and src_masks.shape[1] == 1:
            src_masks = src_masks[:, 0]
        if tgt_masks.ndim == 4 and tgt_masks.shape[1] == 1:
            tgt_masks = tgt_masks[:, 0]

        loss_mask = F.binary_cross_entropy_with_logits(src_masks, tgt_masks, reduction="none")
        loss_mask = loss_mask.flatten(1).mean(1).mean()
        loss_dice = self._sigmoid_dice_loss(src_masks, tgt_masks).mean()
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def loss_diseases(self, outputs: Mapping[str, torch.Tensor], targets: Sequence[Mapping]):
        disease_logits = outputs.get("pred_disease_logits", outputs.get("disease_logits"))
        if disease_logits is None:
            return {"loss_disease": _zero_from_outputs(outputs)}

        if disease_logits.ndim == 3:
            disease_logits = disease_logits.mean(dim=1)
        if disease_logits.ndim != 2:
            raise ValueError("Disease logits must have shape [B, D] or [B, Q, D].")

        supervised_indices = [
            i for i, target in enumerate(targets)
            if _is_disease_supervised(target) and target.get("disease_labels") is not None
        ]
        if not supervised_indices:
            return {"loss_disease": _zero_from_outputs(outputs)}

        supervised_targets = [targets[i] for i in supervised_indices]
        target_tensor = self._stack_disease_targets(
            supervised_targets,
            device=disease_logits.device,
            channels=disease_logits.shape[-1],
        )
        if target_tensor is None:
            return {"loss_disease": _zero_from_outputs(outputs)}

        logits = disease_logits[supervised_indices]
        if logits.shape[0] != target_tensor.shape[0]:
            raise ValueError("Disease logits batch size does not match the number of supervised disease targets.")

        loss_disease = F.binary_cross_entropy_with_logits(logits, target_tensor.float())
        return {"loss_disease": loss_disease}

    def forward(self, outputs: Mapping[str, torch.Tensor], targets: Sequence[Mapping]):
        losses: Dict[str, torch.Tensor] = {}

        box_targets = [t for t in targets if _is_box_supervised(t)]
        if any(name in self.losses for name in ("labels", "boxes", "masks")):
            if box_targets:
                box_batch_indices = [i for i, t in enumerate(targets) if _is_box_supervised(t)]
                box_outputs: Dict[str, torch.Tensor] = {}
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor) and value.shape[:1] == (len(targets),):
                        box_outputs[key] = value[box_batch_indices]
                    else:
                        box_outputs[key] = value
                indices = self.matcher(box_outputs, box_targets)
                if "labels" in self.losses:
                    losses.update(self.loss_labels(box_outputs, box_targets, indices))
                if "boxes" in self.losses:
                    losses.update(self.loss_boxes(box_outputs, box_targets, indices))
                if "masks" in self.losses:
                    losses.update(self.loss_masks(box_outputs, box_targets, indices))
            else:
                zero = _zero_from_outputs(outputs)
                if "labels" in self.losses:
                    losses["loss_ce"] = zero
                if "boxes" in self.losses:
                    losses["loss_bbox"] = zero
                    losses["loss_giou"] = zero
                if "masks" in self.losses:
                    losses["loss_mask"] = zero
                    losses["loss_dice"] = zero

        if "diseases" in self.losses:
            losses.update(self.loss_diseases(outputs, targets))

        return losses
