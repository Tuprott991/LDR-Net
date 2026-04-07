from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .matcher import box_cxcywh_to_xyxy, generalized_box_iou


class LesionDiseaseCriterion(nn.Module):
    """Multi-task loss for lesion localization and disease prediction."""

    def __init__(self, num_lesions: int, matcher: nn.Module, loss_cfg: Dict) -> None:
        super().__init__()
        self.num_lesions = num_lesions
        self.matcher = matcher
        self.loss_cfg = loss_cfg

    def _num_boxes(self, targets: List[dict], device: torch.device) -> Tensor:
        count = sum(target["boxes"].shape[0] for target in targets if bool(target.get("has_boxes", True)))
        return torch.as_tensor(max(count, 1), dtype=torch.float32, device=device)

    def loss_labels(self, outputs: dict, targets: List[dict], indices: List) -> Tensor:
        src_logits = outputs["lesion_logits"]
        batch_size, num_queries, _ = src_logits.shape
        target_classes = torch.full(
            (batch_size, num_queries),
            fill_value=self.num_lesions,
            dtype=torch.int64,
            device=src_logits.device,
        )

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            target_classes[batch_index, src_idx] = targets[batch_index]["labels"][tgt_idx]

        class_weights = torch.ones(self.num_lesions + 1, device=src_logits.device)
        class_weights[-1] = self.loss_cfg["no_object_weight"]
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=class_weights)

    def loss_boxes(self, outputs: dict, targets: List[dict], indices: List, num_boxes: Tensor) -> tuple[Tensor, Tensor]:
        src_boxes = outputs["lesion_boxes"]
        matched_src = []
        matched_tgt = []
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            matched_src.append(src_boxes[batch_index, src_idx])
            matched_tgt.append(targets[batch_index]["boxes"][tgt_idx])

        if not matched_src:
            zero = src_boxes.sum() * 0.0
            return zero, zero

        src = torch.cat(matched_src, dim=0)
        tgt = torch.cat(matched_tgt, dim=0)
        loss_bbox = F.l1_loss(src, tgt, reduction="sum") / num_boxes
        loss_giou = (1.0 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src),
            box_cxcywh_to_xyxy(tgt),
        ))).sum() / num_boxes
        return loss_bbox, loss_giou

    def loss_uncertainty(self, outputs: dict, indices: List) -> Tensor:
        uncertainties = outputs["lesion_uncertainty"]
        matched = []
        for batch_index, (src_idx, _) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            matched.append(uncertainties[batch_index, src_idx])
        if not matched:
            return uncertainties.mean() * 0.0
        return torch.cat(matched, dim=0).mean()

    def loss_disease(self, outputs: dict, targets: List[dict]) -> Tensor:
        disease_logits = outputs["disease_logits"]
        target_vectors = []
        valid_indices = []
        for batch_index, target in enumerate(targets):
            disease_labels = target.get("disease_labels")
            if disease_labels is None or not bool(target.get("has_disease_labels", True)):
                continue
            target_vectors.append(disease_labels.to(device=disease_logits.device, dtype=torch.float32))
            valid_indices.append(batch_index)

        if not target_vectors:
            return disease_logits.mean() * 0.0

        stacked_targets = torch.stack(target_vectors, dim=0)
        stacked_logits = disease_logits[torch.as_tensor(valid_indices, device=disease_logits.device)]
        return F.binary_cross_entropy_with_logits(stacked_logits, stacked_targets)

    def forward(self, outputs: dict, targets: List[dict]) -> Dict[str, Tensor]:
        indices = self.matcher(outputs, targets)
        num_boxes = self._num_boxes(targets, outputs["lesion_logits"].device)

        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices, num_boxes)
        loss_uncertainty = self.loss_uncertainty(outputs, indices)
        loss_disease = self.loss_disease(outputs, targets)

        detection = (
            loss_ce
            + self.loss_cfg["bbox_weight"] * loss_bbox
            + self.loss_cfg["giou_weight"] * loss_giou
            + self.loss_cfg["uncertainty_weight"] * loss_uncertainty
        )
        loss_total = (
            self.loss_cfg["detection_weight"] * detection
            + self.loss_cfg["disease_weight"] * loss_disease
        )

        return {
            "loss_total": loss_total,
            "loss_detection": detection,
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_uncertainty": loss_uncertainty,
            "loss_disease": loss_disease,
        }
